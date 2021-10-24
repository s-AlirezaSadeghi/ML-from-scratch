import numpy as np
from collections import Counter
from scipy.stats import entropy
from typing import List
from math import e
from src.utility.metrics import gini


class SplitCondition:
    def __init__(self, value, col: int, col_names: List):
        self.value = value
        self.col = col
        self.col_names = col_names

    def match(self, data):
        subspace = data[:, self.col]
        if np.issubdtype(type(self.value), np.signedinteger) or isinstance(self.value, float):
            return subspace >= self.value
        elif isinstance(self.value, str):
            return subspace == self.value

    def __repr__(self):
        condition = "=="
        if np.issubdtype(type(self.value), np.signedinteger) or isinstance(self.value, float):
            condition = ">="
        return f"Is {self.col_names[self.col]} {condition} {str(self.value)}"


class ClassificationTree:
    def __init__(self, headers: List, impurity: str = 'gini', min_samples_split: int = 10, max_depth: int = 50):
        self.impurity = impurity
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.headers = headers

    # return distinct values of an array
    def unique_vals(self, rows, col):
        return np.unique(rows[:, col])

    # paritions/splits data based on a condition
    def parition(self, data, condition):
        '''
        splits data based on the given condition into 2 paritions
        :param data: an array of data to be splitted
        :type data: np.array
        :param condition: a condition which determines the how to split data
        :type condition:  SplitCondition class
        :return: two parition of data splitted by condition passed to the method
        :rtype: (np.array,np.array)
        '''
        x, y = data
        true_case = (x[condition.match(x)],y[condition.match(x)])
        false_case = (x[~(condition.match(x))],y[~(condition.match(x))])
        return true_case, false_case

    # https: // www.youtube.com / watch?v = LDRbO9a6XPU
    # https: // machinelearningmastery.com / implement - decision - tree - algorithm - scratch - python /

    def gini(self, data):
        """
        Calculate gini index for the given array
        :param data: an array of values
        :type data: np.array
        :return: gini index of input array
        :rtype: float
        """
        return gini(data)

    #   information gain
    def information_gain(self, left, right, current_uncertainty):
        """
        Calculates the information gain (weighted gini ) based on the splits ( right and left ) compared to the existing uncertainity
        :param left: left split of tree node
        :type left: np.array
        :param right: right split of tree node
        :type right:  np.array
        :param current_uncertainty: gini index of array before splits
        :type current_uncertainty: float
        :return: quantity of infortmation gain by making splits rightr and left based on the condition
        :rtype: float
        """
        # calculate left split  weight - right split weight is 1- left split
        p = float(len(left)) / (len(left) + len(right))
        return current_uncertainty - p * self.gini(left[1]) - (1 - p) * self.gini(right[1])

    def best_split(self, x: np.array, y: np.array):
        """
        determines the best condition to split a node into two paritions using gini and information gain iteratively
        :param data: node data (whether root or sub-root )
        :type data: np.ndarray
        :return: a condition/question  based on features and their distinct values which leads to highest information gain after split to
        left and right split
        :rtype: (float,SplitCondition)
        """
        best_gain = 0
        best_cond = None
        # Calculate gini score before split
        current_uncertainty = self.gini(y)

        # Excluding targer var from n_features
        n_features = x.shape[1]

        # find split condition that increase information_gain
        for col in range(n_features):
            distinct_vals = Counter(x[:, col])
            for val in distinct_vals:
                # split condition
                cond = SplitCondition(val, col, self.headers)
                # Partition data and return Tuple of split in (x,y) format
                true_case, false_case = self.parition((x, y), cond)

                if len(true_case[0]) == 0 or len(false_case[0]) == 0:
                    continue

                gain = self.information_gain(true_case, false_case, current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_cond = gain, cond
        return best_gain, best_cond

    def fit(self, x: np.array, y: np.array):

        # get best split with most gain
        gain, condition = self.best_split(x, y)

        if gain == 0:
            return Leaf(x, y)

        true_data, false_data = self.parition((x, y), condition)

        # Recursively build the true branch.
        left_branch = self.tree_builder(true_data)

        # Recursively build the false branch.
        right_branch = self.tree_builder(false_data)

        return Decision_Node(condition, left_branch, right_branch)


class Leaf:
    """A Leaf node classifies data.
    Contains a dictionary of target class(es) (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, data):
        self.predictions = Counter(data[:, -1])


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

    # def predict(self,data)


# def tree_builder(data: np.array, labels: List):
#     ct = ClassificationTree(headers=labels)
#     # get best split with most gain
#     gain, condition = ct.best_split(data)
#
#     if gain == 0:
#         return Leaf(data)
#
#     true_data, false_data = ct.parition(data, condition)
#
#     # Recursively build the true branch.
#     left_branch = tree_builder(true_data)
#
#     # Recursively build the false branch.
#     right_branch = tree_builder(false_data)
#
#     return Decision_Node(condition, left_branch, right_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")




def entropy_scipy(data):
    labels, label_freqs = np.unique(data, return_counts=True)
    probabilities = label_freqs / label_freqs.sum()
    return entropy(probabilities)


def entropy_numpy(data):
    labels, label_freqs = np.unique(data, return_counts=True)
    probabilities = label_freqs / label_freqs.sum()
    return -(probabilities * np.log(probabilities) / np.log(e)).sum()




def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

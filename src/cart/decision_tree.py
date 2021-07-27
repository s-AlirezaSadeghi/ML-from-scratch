import numpy as np
from collections import Counter
from typing import List


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def gini(data):
    labels = Counter(data)
    impurity = 1
    for label in labels:
        # go over each label in counts
        prob = labels[label] / float(len(data))
        impurity -= prob ** 2
    return impurity


def information_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return  current_uncertainty - p * gini(left[:, -1]) - (1 - p) * gini(right[:,-1])

# def gini(arr):
#     array = arr.flatten()
#     array = array.astype('float64')
#     if np.amin(array) < 0:
#         # Values cannot be negative:
#         array -= np.amin(array)
#     # Values cannot be 0:
#     array += 0.0000001
#     # Values must be sorted:
#     array = np.sort(array)
#     # Index per array element:
#     index = np.arange(1, array.shape[0] + 1)
#     # Number of array elements:
#     n = array.shape[0]
#     # Gini coefficient:
#     return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

#https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
#https://github.com/random-forests/tutorials/blob/master/decision_tree.py
#https://github.com/random-forests/tutorials/blob/master/decision_tree.ipynb
#https://www.youtube.com/watch?v=LDRbO9a6XPU
#https://github.com/oliviaguest/gini/blob/master/gini.py

class SplitCondition:
    def __init__(self, value, col: int, col_names: List):
        self.value = value
        self.col = col
        self.col_names = col_names

    def match(self, data):
        subspace = data[:, self.col]
        if np.issubdtype(type(self.value),int) or isinstance(self.value, float):
            return subspace >= self.value
        elif isinstance(self.value,str):
            return subspace == self.value

    def __repr__(self):
        condition = "=="
        if np.issubdtype(type(self.value),int) or isinstance(self.value, float):
            condition = ">="
        return f"Is {self.col_names[self.col]} {condition} {str(self.value)}"


class ClassificationTree:
    def __init__(self,headers: List, impurity: str = 'gini', min_samples_split: int = 10, max_depth: int = 50 ):
        self.impurity = impurity
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.headers= headers


    #Get distinct values of an array
    def unique_vals(self, rows, col):
        return np.unique(rows[:, col])

    #parition data based on a condition
    def parition(self,data,condition):
        true_case = data[condition.match(data)]
        false_case = data[~(condition.match(data))]
        return true_case, false_case

    def gini(self,data):
        labels = Counter(data)
        impurity = 1
        for label in labels:
            # go over each label in counts
            prob = labels[label] / float(len(data))
            impurity -= prob ** 2
        return impurity

    #   information gain
    def information_gain(self,left,right,current_uncertainty):
        p = float(len(left)) / (len(self.left) + len(self.right))
        return current_uncertainty - p * gini(left[:, -1]) - (1 - p) * gini(right[:, -1])



    def best_split(self,data):
        best_gain = 0
        best_cond = None
        current_uncertainty = self.gini(data[:,-1])
        n_features = data.shape[1]-1
        for col in range(n_features):
            distinct_vals = Counter(data[:,col])
            for val in distinct_vals:
                cond = SplitCondition(val,col,self.headers)
                true_case , false_case = self.parition(data,cond )

                if len(true_case) == 0 or len(false_case) == 0:
                    continue

                gain = information_gain(true_case,false_case,current_uncertainty)

                if gain >= best_gain:
                    best_gain, best_cond = gain, cond
        return best_gain, best_cond


#sample data creation
cardiac_data = np.random.randint(19, 150, size=(45, 2),dtype=int)
heart_attack = np.random.choice([0, 1], size=(45, 1), p=[0.90, 0.10])
training_data = np.column_stack((cardiac_data,heart_attack))

# calculate starting current_uncertainty
current_uncertainty = gini(training_data[:,-1])
print(f'current_uncertainty is : {current_uncertainty}')
print(Counter(training_data[:,-1]))


# for _ in range(1,2):
#     cond = SplitCondition(147,0,['heart_rate','age','heart_attack'])
#     print(cond.__repr__())
#
#     true_case = training_data[cond.match(training_data)]
#     false_case = training_data[~(cond.match(training_data))]
#     print(f'info gain is {information_gain(true_case, false_case, current_uncertainty)}')
#
# true_case.shape
# false_case.shape
ct = ClassificationTree(headers=['heart_rate','age','heart_attack'])

ct.best_split(training_data)

class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, data):
        self.predictions = Counter(data[:,-1])



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

def tree_builder(data):

    ct = ClassificationTree(headers=['heart_rate', 'age', 'heart_attack'])
    # get best split with most gain
    gain , condition = ct.best_split(data)

    if gain == 0:
        return Leaf(data)

    true_data, false_data = ct.parition(data, condition)

    # Recursively build the true branch.
    true_branch = tree_builder(true_data)

    # Recursively build the false branch.
    false_branch = tree_builder(false_data)

    return Decision_Node(condition, true_branch, false_branch)


my_tree = tree_builder(training_data)

def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

print_tree(my_tree)

print('Job done!')
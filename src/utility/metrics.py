import numpy as np
from scipy import stats
from scipy.spatial import distance
from typing import List


def mse(y_hat: np.ndarray, y_true: np.ndarray):
    assert isinstance(y_hat, np.ndarray), "predictions should be numpy array"
    assert isinstance(y_true, np.ndarray), "True values should be numpy array"
    assert len(y_true) == len(y_hat), "Lenght of Y pred and Y True don't match!"
    return ((y_true - y_hat) ** 2).sum() / len(y_true)


def rmse(y_hat: np.ndarray, y_true: np.ndarray):
    assert isinstance(y_hat, np.ndarray), "predictions should be numpy array"
    assert isinstance(y_true, np.ndarray), "True values should be numpy array"
    assert len(y_true) == len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.sqrt(np.mean((y_hat - y_true) ** 2))


def mae(y_hat: np.ndarray, y_true: np.ndarray):
    assert isinstance(y_hat, np.ndarray), "predictions should be numpy array"
    assert isinstance(y_true, np.ndarray), "True values should be numpy array"
    assert len(y_true) == len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.average(np.abs(y_hat - y_true))


def cross_entropy(y_hat: np.ndarray, y_true: np.ndarray, epsilon=1e-12):
    assert isinstance(y_hat, np.ndarray), "predictions should be numpy array"
    assert isinstance(y_true, np.ndarray), "True values should be numpy array"
    assert len(y_true) == len(y_hat), "Lenght of Y pred and Y True don't match!"
    assert (y_hat >= 0).all() and (y_hat <= 1).all(), 'Y pred should be between 0 and 1'
    # todo: assert whether target is either 0-1 values
    preds = np.clip(y_hat, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(preds + 1e-9)) / len(y_hat)


def accuracy(y_hat: np.ndarray, y_true: np.ndarray):
    assert isinstance(y_hat, np.ndarray), "predictions should be numpy array"
    assert isinstance(y_true, np.ndarray), "True values should be numpy array"
    assert len(y_true) == len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.sum(y_true == y_hat) / len(y_true)


# Todo : finish this
def logloss():
    """
    Log loss, aka logistic loss or cross-entropy loss.


    :return:
    """
    ...


def gini(data: List[str]) -> float:
    """
    This function calculates Gini score:
     it gives an idea of how good a split is by
     how mixed the classes are in the two groups created by the split.
     A perfect separation results in a Gini score of 0,
     whereas the worst case split that results in 0.5 (50/50) in case of discreet classes
    :param data: a list of labels
    :type data: List
    :return: A Gini Score
    :rtype: float
    """
    labels, label_freqs = np.unique(data, return_counts=True)
    return 1 - (np.power(label_freqs / label_freqs.sum(), 2)).sum()


# Todo : finish this
def partial_derivative():
    ...


def entropy(data: list):
    """Calculates entropy of the  `List` passed
        """
    assert isinstance(data, list), "Data should be of List type"
    value, counts = np.unique(data, return_counts=True)  # counts occurrence of each value
    return stats.entropy(counts, base=2)  # get entropy from counts


# todo: finish confusion confusion_matrix
def confusion_matrixx():
    ...


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> List:
    """
    manhatan distance : Square root of sum of squares differences between points
    :param a:
    :type a:
    :param b:
    :type b:
    :return:
    :rtype:
    """
    return list(map(distance.euclidean, a, b))


# todo: write manhatan distance function
def manhatan_distance(a: np.ndarray, b: np.ndarray) -> List:
    ...


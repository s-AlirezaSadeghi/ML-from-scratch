import pytest
import numpy as np
from src.utility.metrics import entropy
from src.utility.metrics import mse
from src.utility.metrics import gini
from src.utility.metrics import euclidean_distance


def test_dumb_sum():
    assert sum([1, 2, 3]) == 6, "Should be 6"


@pytest.mark.parametrize("test_labels, expected", [(["SAM", "SAM"], 0),
                                                   (["SAM", "ALI"], 1),
                                                   (["SAM", "SAM", "ALI"], 0.9182958340544894)])
def test_entropy(test_labels, expected):
    assert entropy(test_labels) == expected, f"Should be {expected}"


@pytest.mark.parametrize("test_yhat, test_ytrue, expected",
                         [(np.array([2, 2, 2]), (np.array([1.9, 1.8, 1.7])), 0.04666666666666667),
                          (np.array([2, 2, 2]), (np.array([1.9, 1.8, 1.7])), 0.04666666666666667), ])
def test_mse(test_yhat, test_ytrue, expected):
    assert mse(test_yhat, test_ytrue) == expected, f"Expected value is {expected}"


@pytest.mark.parametrize("test_labels, expected", [(["SAM", "SAM"], 0),
                                                   (["SAM", "ALI"], 0.5),
                                                   ]
                         )
def test_gini(test_labels, expected):
    assert gini(test_labels) == expected, f"Expected value is {expected}"


@pytest.mark.parametrize("a, b, expected", [
    (np.array([[0.92133559, 0.79611501], [0.73425432, 0.47863958]]),
     np.array([[0.89599036, 0.35331696], [0.57095202, 0.4116479]]),
     [0.44352282214961086, 0.17650928127017118]), ])
def test_euclidean_distance(a, b, expected):
    assert euclidean_distance(a, b) == expected, f"Expected value is {expected}"

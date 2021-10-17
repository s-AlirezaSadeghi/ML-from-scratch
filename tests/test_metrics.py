import pytest
import numpy as np
from src.utility.metrics import entropy, mse


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

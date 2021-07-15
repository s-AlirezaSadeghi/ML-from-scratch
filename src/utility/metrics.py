import numpy as np



def mse(y_hat, y_true):
    assert len(y_true) ==len(y_hat), "Lenght of Y pred and Y True don't match!"
    return ((y_true - y_hat) ** 2).sum() / len(y_true)

def rmse(y_hat, y_true):
    assert len(y_true) ==len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.sqrt(np.mean((y_hat - y_true) ** 2))


def mae(y_hat, y_true):
    assert len(y_true) ==len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.average(np.abs(y_hat - y_true))

def cross_entropy(y_hat,y_true,epsilon=1e-12):

    assert len(y_true) ==len(y_hat), "Lenght of Y pred and Y True don't match!"
    assert (y_hat >= 0).all() and (y_hat <= 1).all() , 'Y pred should be between 0 and 1'
    #todo: assert whether target is either 0-1 values
    preds = np.clip(y_hat, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(preds + 1e-9)) / len(y_hat)

def accuracy(y_hat,y_true):
    assert len(y_true) ==len(y_hat), "Lenght of Y pred and Y True don't match!"
    return np.sum(y_true==y_hat)/len(y_true)

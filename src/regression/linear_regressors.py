import numpy as np
from abc import ABC, abstractmethod
from src.utility.metrics import mse, rmse, cross_entropy
import logging


class Regressor(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def predict(self):
        pass


class LinearRegressor(Regressor):
    def __init__(self, learning_rate: float = 0.005, iterations: int = 500, verbose: bool = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.vebose = verbose
        self.cost_tracker = []
        self.rsme_tracker = []
        self.bias, self.weights = None, None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        :param X: This a dataframe with all the independent variables
        :type X: np.ndarray
        :param Y: This is an array of target/dependent variables
        :type Y: np.ndarray
        :return: self
        :rtype: LinearRegressor object
        """
        self.m, self.n = X.shape
        self.X = X
        self.Y = Y
        self.bias = 1
        self.weights = np.zeros(self.n)

        # todo: assert x and y shapes

        try:
            for _ in range(self.iterations):
                self.update_bias_weight()
                loss = self._mse(self.X, self.Y)
                rmse_loss = self._rmse(self.X, self.Y)
                if self.vebose:
                    print(f'Current RMSE is {rmse_loss} @ {_}')
                self.cost_tracker.append((loss, _))
                self.rsme_tracker.append((rmse_loss, _))
        except Exception as e:
            print(e)
            return self

        return self

    def update_bias_weight(self):
        '''
        This method update Weights and biases using  gradient descent
        :return: self
        :rtype: LinearRegressor object
        '''
        self.dW, self.dbias = self.calculate_partial_derivative()
        self.weights = self.weights - (self.learning_rate * self.dW)
        self.bias = self.bias - (self.learning_rate * self.dbias)

        return self

    def calculate_partial_derivative(self):
        '''
        This method calculate the partial derivative for weights and bias
        :return: self
        :rtype: LinearRegressor object
        '''
        Y_hat = self.predict(self.X)

        dWeights = (2 * (self.X.T).dot(Y_hat - self.Y)) / self.m
        dBias = 2 * np.sum(Y_hat - self.Y) / self.m

        return dWeights, dBias

    def predict(self, X):
        """
        This method use leared theta/weights and bias to make predictions
        :param X: Independept variables
        :type X: np.ndarray
        :return: Prediction values
        :rtype: np.ndarray
        """
        return X.dot(self.weights) + self.bias

    def _mse(self, X, Y):
        """
        This method calculate mean squared error of given X and Y for weights and bias of model
        :param X: This a dataframe with all the independent variables
        :type X: np.ndarray
        :param Y:
        :type Y: np.ndarray
        :return: mean squared error
        :rtype: np.float64
        """
        return mse(self.predict(X), Y)

    def _rmse(self, X, Y):
        """
        This method calculates root mean squared error of given X and Y for weights and bias of model

        :param X: This a dataframe with all the independent variables
        :type X: np.ndarray
        :param Y: This is an array of target/dependent variables
        :type Y: np.ndarray
        :return: Root mean suqred error
        :rtype: np.float64
        """
        return rmse(self.predict(X), Y)



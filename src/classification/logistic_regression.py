import numpy as np
from src.regression.linear_regressors import Regressor
import logging
from src.utility.metrics import cross_entropy

class LogisticRegression(Regressor):
    def __init__(self, learning_rate: float = 0.005, iterations: int = 500, threshold: np.float16 = 0.5,
                 verbose: bool = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.vebose = verbose
        self.loss_tracker = []
        self.threshold = threshold
        self.bias, self.weights = None, None

    def _sigmoid(self, y_hat):
        return 1.0 / (1 + np.exp(-y_hat))

    def predict(self, X):
        y_hat = self._sigmoid((X.dot(self.weights) + self.bias))
        return np.where(y_hat >= self.threshold, 1, 0)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        '''
        :param X: This a dataframe with all the independent variables
        :type X: np.ndarray
        :param Y: This is an array of target/dependent variables
        :type Y: np.ndarray
        :return: self
        :rtype: LinearRegressor object
        '''
        self.m, self.n = X.shape
        self.X = X
        self.Y = Y
        self.bias = 1
        self.weights = np.zeros(self.n)

        # todo: assert x and y shapes

        try:
            for _ in range(self.iterations):
                self.update_bias_weight()
                y_hat = self.predict(self.X)
                loss = self.loss(y_hat, self.Y)
                if self.vebose:
                    print(f'Current loss is {loss} @ {_}')
                    logging.debug(f'Current loss is {loss} @ {_}')
                self.loss_tracker.append((loss, _))
        except Exception as e:
            print(e)
            return self

        return self

    def loss(self, Y_hat, Y):
        return cross_entropy(Y_hat, Y)

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

        dWeights = (1 / self.m) * (self.X.T).dot(Y_hat - self.Y)
        dBias = (1 / self.m) * np.sum(Y_hat - self.Y)

        return dWeights, dBias

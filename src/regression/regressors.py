import numpy as np
import pandas as pd

class LinearRegressor:
    def __init__(self, learning_rate: int = 0.01, iterations: int = 500):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self,X:pd.DataFrame,Y):
        '''
        :param X: This a dataframe with all the independent variables
        :type X: Panda.DataFrame
        :param Y: This is an array of
        :type Y:
        :return:
        :rtype:
        '''

        self.m , self.n = X.shape
        self.bias = 1
        self.W = np.ones(self.n)
        self.X = X
        self.Y = Y
        self.error=[]


        #todo: assert x and y

        for _ in range(self.iterations):
            self.update_bias_weight()
        return self

    def update_bias_weight(self):
        '''
        This method update Weights and biases based on gradient and learning rate
        :return: self
        :rtype: LinearRegressor object
        '''
        Y_hat = self.predict(self.X)
        self.error.append(self.mse())
        self.dW , self.dbias = self.calculate_gradient(Y_hat)

        self.W = self.W - self.learning_rate * self.dW
        self.bias = self.bias - self.learning_rate * self.dbias
        return self

    def calculate_gradient(self, Y_hat):
        '''
        This method calculate the gradients of weights and biases
        :param Y_hat:
        :type Y_hat:
        :return: self
        :rtype: LinearRegressor object
        '''
        dWeights = - (2 * (self.X.T).dot(self.Y - Y_hat)) / self.m
        dBias = - 2 * np.sum(self.Y - Y_hat) / self.m
        return dWeights, dBias

    def predict(self, X):
        return X.dot(self.W)+self.bias


    def mse(self):
        return np.square(self.Y - self.predict(self.X)).mean()


#!/usr/bin/env python3
"""module that define a neural network
"""

import numpy as np


class NeuralNetwork:
    """
    class that defines a neural network with one hidden
    layer performing binary classification
    """
    def __init__(self, nx, nodes):
        """class constructor
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """method to get the weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """method to get the bias for the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """method to get the activated output for the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """method to get the weights vector for the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """method to get the bias for the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """method to get the activated output for the output neuron.
        """
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network
        """
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression
        """
        loss1 = -np.matmul(Y, np.log(A).T)
        loss2 = np.matmul((1 - Y), np.log(1.0000001 - A).T)
        cost = loss1 - loss2
        cost = np.sum(cost) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network’s predictions
        """
        _, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network
        """
        print(X.shape, Y.shape, A1.shape, A2.shape)
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.T) / X.shape[1]
        db2 = np.sum(dz2, axis=1, keepdims=True) / X.shape[1]
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
        f1 = np.matmul(self.__W2.T, dz2)
        derivative = np.matmul(A1.T, (1 - A1))
        dz1 = np.matmul(f1, derivative)
        dw1 = np.matmul(dz1, X.T) / X.shape[1]
        db1 = np.sum(dz1, axis=1, keepdims=True) / X.shape[1]
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        

#!/usr/bin/env python3
"""module that defines a deep neural network
"""

import numpy as np


class DeepNeuralNetwork:
    """
    class that defines a deep neural network
    performing binary classification
    """

    def __init__(self, nx, layers):
        """class constructor
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        l_prev = nx
        for l in range(len(layers)):
            if not isinstance(layers[l], int):
                raise TypeError('layers must be a list of positive integers')
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            self.__weights[key] = np.random.randn(
                layers[l], l_prev) * np.sqrt(2 / l_prev)
            self.__weights[bias] = np.zeros((layers[l], 1))
            l_prev = layers[l]

    @property
    def L(self):
        """get number of layers in the neural network.
        """
        return self.__L

    @property
    def cache(self):
        """get dictionary to hold all intermediary values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """get dictionary to hold all weights and biased of the network.
        """
        return self.__weights

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for l in range(self.__L):
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            w = self.__weights[key]
            key_cache = 'A{}'.format(l)
            cache = self.__cache[key_cache]
            z = np.matmul(w, cache) + self.__weights[bias]
            A = 'A{}'.format(l + 1)
            self.__cache[A] = 1 / (1 + np.exp(-z))
        out = 'A{}'.format(self.__L)
        return self.__cache[out], self.__cache

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression
        """
        loss1 = np.matmul(Y, np.log(A).T)
        loss2 = np.matmul(1 - Y, np.log(1.0000001 - A.T))
        m = Y.shape[1]
        cost = np.sum(-(loss1 + loss2)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluate the neural network’s predictions
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network
        """
        for layer in range(self.__L, 0, -1):
            key = 'A{}'.format(layer)
            input = 'A{}'.format(layer - 1)
            w = 'W{}'.format(layer)
            out = 'A{}'.format(self.__L)
            bias = 'b{}'.format(layer)
            if layer == self.__L:
                dz = cache[out] - Y
                dw = np.matmul(dz, cache[input].T) / Y.shape[1]
            else:
                w1 = 'W{}'.format(layer + 1)
                back = np.matmul(self.__weights[w1].T, dz)
                derivative = cache[key] * (1 - cache[key])
                dz = back * derivative
                dw = np.matmul(dz, cache[input].T) / Y.shape[1]
            db = np.sum(dz, axis=1, keepdims=True) / Y.shape[1]
            self.__weights[w] = self.__weights[w] - alpha * dw
            self.__weights[bias] = self.__weights[bias] - alpha * db
        return self.__weights

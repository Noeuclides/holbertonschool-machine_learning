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
        """
        - nx: number of input features
        - layers: list representing the number of nodes in
        each layer of the network
        - L: number of layers in the neural network.
        - cache: dictionary to hold all intermediary values of the network.
        - weights: dictionary to hold all weights and biased of the network.
        Initialized using the He et al. method and saved in the weights
        dictionary using the key W{l} where {l} is the hidden layer
        the weight belongs to.
        - biases of the network are initialized to 0’s and saved
        in the weights dictionary using the key b{l} where {l} is
        the hidden layer the bias belongs to
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
        """
        Calculates the forward propagation of the neural network
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        - m: number of examples
        Updates the private attribute __cache:
        The activated outputs of each layer are saved in the __cache
        dictionary with the key A{l} where {l} is the hidden layer
        the activated output belongs to
        X are saved in the cache dictionary using the key A0
        All neurons use a sigmoid activation function
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
        """
        Calculates the cost of the model using logistic regression
        - Y: numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data
        - A: numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Returns the cost
        """
        loss1 = np.matmul(Y, np.log(A).T)
        # 1.0000001 - A instead of 1 - A to avoid division by zero errors
        loss2 = np.matmul(1 - Y, np.log(1.0000001 - A.T))
        m = Y.shape[1]
        cost = np.sum(-(loss1 + loss2)) / m
        return cost

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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        l_prev = nx
        for l in range(len(layers)):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError('layers must be a list of positive integers')
            key = 'W{}'.format(l + 1)
            bias = 'b{}'.format(l + 1)
            self.weights[key] = np.random.randn(
                layers[l], l_prev) * np.sqrt(2 / l_prev)
            self.weights[bias] = np.zeros((layers[l], 1))
            l_prev = layers[l]

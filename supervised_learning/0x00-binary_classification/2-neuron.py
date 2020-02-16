#!/usr/bin/env python3
"""module that defines a single neuron
"""
import numpy as np


class Neuron:
    """define a single neuron performing binary classification
    """
    def __init__(self, nx):
        """class constructor
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """setter method of the weights
        """
        return self.__W

    @property
    def b(self):
        """setter method of the bias
        """
        return self.__b

    @property
    def A(self):
        """setter method of the activated output
        """
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron
        """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

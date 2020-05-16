#!/usr/bin/env python3
"""module that defines a single neuron
"""
import numpy as np


class Neuron:
    """define a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        - nx: number of input features to the neuron

        Private instance attributes:
        -__W: The weights vector for the neuron.
        Initialized using a random normal distribution.
        - __b: The bias for the neuron.
        Initialized to 0.
        - __A: The activated output of the neuron (prediction).
        Initialized to 0.
        Each private attribute have its getter function
        (no setter function).
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
        """
        Calculates the forward propagation of the neuron
        - X: numpy.ndarray with shape (nx, m) that contains the input data
        - nx: number of input features to the neuron
        - m: number of examples
        Updates the private attribute __A
        The neuron use a sigmoid activation function
        - sig(z) = 1 / (1 + exp(-z))
        Returns the private attribute __A
        """
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Loss Function:
        L(A, Y) = -(Y * log(A) + (1 - Y) * log(1 - A)))
        Cost function:
        J(w, b) = (1 / m) * ∑(i=1; i<=m) L(A, Y)
        - Y: numpy.ndarray with shape (1, m) that contains the correct
        labels for the input data.
        - A: numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Returns the cost
        """
        loss1 = np.matmul(Y, np.log(A).T)
        # 1.0000001 - A instead of 1 - A to avoid division by zero errors
        loss2 = np.matmul(1 - Y, np.log(1.0000001 - A).T)
        m = Y.shape[1]
        cost = np.sum(-(loss1 + loss2)) / m
        return cost

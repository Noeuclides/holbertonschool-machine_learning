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

        Public instance attributes:
        - W: The weights vector for the neuron.
        Initialized using a random normal distribution.
        - b: The bias for the neuron. Initialized to 0.
        - A: The activated output of the neuron (prediction).
        Initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0

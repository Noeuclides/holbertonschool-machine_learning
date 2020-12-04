#!/usr/bin/env python3
"""
bidirectional module
"""
import numpy as np


class BidirectionalCell:
    """
    bidirectional class
    """
    def __init__(self, i, h, o):
        """
        class constructor

        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy,
        bhf, bhb, by that represent the weights and biases of the cell
            - Whf and bhf are for the hidden states in the
            forward direction
            - Whb and bhb are for the hidden states in the
            backward direction
            - Wy and by are for the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction
        for one time step

        - x_t: numpy.ndarray of shape (m, i) that contains the
        data input for the cell
        - m: batch size for the data
        - h_prev: numpy.ndarray of shape (m, h) containing the
        previous hidden state
        Returns: h_next, the next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction
        for one time step

        - x_t: numpy.ndarray of shape (m, i) that contains the
        data input for the cell
        - m: batch size for the data
        - h_next: numpy.ndarray of shape (m, h) containing the
        next hidden state
        Returns: h_pev, the previous hidden state
        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_back = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_back

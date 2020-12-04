#!/usr/bin/env python3
"""
GRU module
"""
import numpy as np


class GRUCell:
    """
    GRU class
    """
    def __init__(self, i, h, o):
        """
        class constructor

        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        Creates the public instance attributes Wz, Wr, Wh, Wy,
        bz, br, bh, by that represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """
        sigmoid activation function
        """
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        """
        performs softmax values for each sets of scores in x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        - x_t: numpy.ndarray of shape (m, i) that contains the
        data input for the cell
        - m: batche size for the data
        - h_prev: numpy.ndarray of shape (m, h) containing the
        previous hidden state

        Returns: h_next, y
            - h_next: next hidden state
            - y: output of the cell
        """
        h_x_mat = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(h_x_mat, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(h_x_mat, self.Wr) + self.br)

        r_x_mat = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_1 = np.tanh(np.matmul(r_x_mat, self.Wh) + self.bh)
        h_next = z_gate * h_1 + (1 - z_gate) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

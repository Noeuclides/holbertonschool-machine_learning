#!/usr/bin/env python3
"""
LSTM module
"""
import numpy as np


class LSTMCell:
    """
    LSTM class
    """
    def __init__(self, i, h, o):
        """
        class constructor

        - i: dimensionality of the data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo,
        Wy, bf, bu, bc, bo, by that represent the weights and
        biases of the cell
            - Wf and bf are for the forget gate
            - Wu and bu are for the update gate
            - Wc and bc are for the intermediate cell state
            - Wo and bo are for the output gate
            - Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step

        - x_t: numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        - m: batche size for the data
        - h_prev: numpy.ndarray of shape (m, h) containing the
        previous hidden state
        - c_prev: numpy.ndarray of shape (m, h) containing the
        previous cell state
        Returns: h_next, c_next, y
            - h_next: next hidden state
            - c_next: next cell state
            - y: output of the cell
        """
        h_x_mat = np.concatenate((h_prev, x_t), axis=1)
        u_gate = self.sigmoid(np.matmul(h_x_mat, self.Wu) + self.bu)
        f_gate = self.sigmoid(np.matmul(h_x_mat, self.Wf) + self.bf)
        o_gate = self.sigmoid(np.matmul(h_x_mat, self.Wo) + self.bo)

        c_1 = np.tanh(np.matmul(h_x_mat, self.Wc) + self.bc)
        c_next = u_gate * c_1 + f_gate * c_prev
        h_next = o_gate * np.tanh(c_next)

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y

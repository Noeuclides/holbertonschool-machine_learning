#!/usr/bin/env python3
"""
rnn module
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    performs forward propagation for a simple RNN:

    - rnn_cell: instance of RNNCell that will be used for the
    forward propagation
    - X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of the data
    - h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
        - h: dimensionality of the hidden state
    Returns: H, Y
        - H: numpy.ndarray containing all of the hidden states
        - Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    o = rnn_cell.by.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for i in range(t):
        h_next, y_next = rnn_cell.forward(H[0], X[i])
        H[i + 1] = h_next
        Y[i] = y_next

    return H, Y

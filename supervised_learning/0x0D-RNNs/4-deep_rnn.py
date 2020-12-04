#!/usr/bin/env python3
"""
Deep RNN module
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    performs forward propagation for a deep RNN:

    - rnn_cells: list of RNNCell instances of length l that will
    be used for the forward propagation
        - l: number of layers
    - X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of the data
    - h_0: initial hidden state, given as a numpy.ndarray
    of shape (l, m, h)
        - h: dimensionality of the hidden state
    Returns: H, Y
        - H: numpy.ndarray containing all of the hidden states
        - Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    _, o = rnn_cells[-1].by.shape
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        h_aux = X[step]
        for layer in range(len(rnn_cells)):
            r_cell = rnn_cells[layer]
            x_t = h_aux
            h_prev = H[step][layer]
            h_next, y_next = r_cell.forward(h_prev=h_prev, x_t=X[step])
            h_aux = h_next
            H[step + 1][layer] = h_aux
        Y[step] = y_next

    return H, Y

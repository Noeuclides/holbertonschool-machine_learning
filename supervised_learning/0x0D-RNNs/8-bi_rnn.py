#!/usr/bin/env python3
"""
bidirectional module
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:

    - bi_cell: instance of BidirectinalCell that will be used
    for the forward propagation
    - X: data to be used, given as a numpy.ndarray of shape (t, m, i)
        - t: maximum number of time steps
        - m: batch size
        - i: dimensionality of the data
    - h_0: initial hidden state in the forward direction,
    given as a numpy.ndarray of shape (m, h)
        - h: dimensionality of the hidden state
    - h_t: initial hidden state in the backward direction,
    given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
        - H: numpy.ndarray containing all of the concatenated hidden states
        - Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    H_forw = np.zeros((t, m, h))
    H_back = np.zeros((t, m, h))
    H_forw[0] = h_0
    H_back[-1] = h_t
    for step in range(t):
        H_for[step + 1] = bi_cell.forward(H_forw[step], X[t])
        H_back[t - 1 - step] = bi_cell.backward(
            H_back[t - step], X[t - 1 - step])
    H = np.concatenate((H_forw, H_back), axis=-1)
    Y = bi_cell.output(H)

    return H, Y

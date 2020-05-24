#!/usr/bin/env python3
"""Shuffle data
"""

import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way
    - X: first numpy.ndarray of shape (m, nx) to shuffle
        - m: number of data points
        - nx: number of features in X
    - Y: second numpy.ndarray of shape (m, ny) to shuffle
        - m: same number of data points as in X
        - ny: number of features in Y
    Returns: the shuffled X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle], Y[shuffle]

#!/usr/bin/env python3
"""Shuffle data
"""

import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way
    """
    shuffle = np.random.permutation(len(X))
    return X[shuffle], Y[shuffle]

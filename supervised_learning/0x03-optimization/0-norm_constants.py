#!/usr/bin/env python3
"""module to normalize constants
"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization)
    constants of a matrix:
    - X: numpy.ndarray of shape (m, nx) to normalize
        - m: number of data points
        - nx: number of features
    Returns: the mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std

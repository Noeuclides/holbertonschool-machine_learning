#!/usr/bin/env python3
"""module to normalize
"""

import numpy as np


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix
    - X: numpy.ndarray of shape (d, nx) to normalize
        - d: number of data points
        - nx: number of features
    - m: numpy.ndarray of shape (nx,) containing the mean of all features of X
    - s: numpy.ndarray of shape (nx,) containing the std of all features of X
    Returns: The normalized X matrix
    """
    norm = (X - m) / s
    return norm

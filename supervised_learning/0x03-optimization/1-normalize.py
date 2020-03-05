#!/usr/bin/env python3
"""module to normalize
"""

import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix
    """
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = (X[i][j] - m[j]) / s[j]
    return X

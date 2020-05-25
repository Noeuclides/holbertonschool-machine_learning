#!/usr/bin/env python3
"""
module to batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network using BN:

    - Z: numpy.ndarray of shape (m, n) that should be normalized
        - m: number of data points
        - n: number of features in Z
    - gamma: numpy.ndarray of shape (1, n) containing the scales used for BN
    - beta: numpy.ndarray of shape (1, n) containing the offsets used for BN
    - epsilon: small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    mean = np.sum(Z, axis=0) / Z.shape[0]
    variance = np.sum((Z - mean)**2, axis=0) / Z.shape[0]

    Znorm = (Z - mean) / (variance + epsilon) ** (1/2)

    batch_norm = gamma * Znorm + beta

    return batch_norm

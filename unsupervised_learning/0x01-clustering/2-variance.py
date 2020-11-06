#!/usr/bin/env python3
"""
Module to get the variance
"""
import numpy as np


def variance(X, C):
    """
    calculates the total intra-cluster variance for a data set:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - C: numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster

    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.size < 1 or C.size < 1:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    if C.shape[0] >= X.shape[0]:
        return None

    distance = np.linalg.norm(X - C[:, None], axis=-1)
    cluster = np.min(distance, axis=0)

    return np.sum(np.square(cluster))

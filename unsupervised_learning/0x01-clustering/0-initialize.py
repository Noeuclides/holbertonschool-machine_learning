#!/usr/bin/env python3
"""
Modulo to initialize centroids
"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:

    - X: numpy.ndarray of shape (n, d) containing the dataset that
    will be used for K-means clustering
        - n: number of data points
        - d: number of dimensions for each data point
    - k: positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    print(X.shape)
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    max = np.max(X, axis=0)
    min = np.min(X, axis=0)

    return np.random.uniform(min, max, size=(k, X.shape[1]))

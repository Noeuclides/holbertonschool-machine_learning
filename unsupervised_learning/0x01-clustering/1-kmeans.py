#!/usr/bin/env python3
"""
Modulo to perform K-means
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
    max = np.max(X, axis=0)
    min = np.min(X, axis=0)

    return np.random.uniform(min, max, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset:

    - X: numpy.ndarray of shape (n, d) containing the dataset
        - n: number of data points
        - d: number of dimensions for each data point
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number of
    iterations that should be performed
    Returns: C, clss, or None, None on failure
        - C: numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        - clss: numpy.ndarray of shape (n,) containing the index of
        the cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    C = initialize(X, k)
    clss = None
    for i in range(iterations):
        distance = np.linalg.norm(X[:, None] - C, axis=-1)
        clss = np.argmin(distance, axis=-1)
        cluster = np.copy(C)
        for j in range(k):
            index = np.argwhere(clss == j)
            if not len(index):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[index], axis=0)
        if (cluster == C).all():
            return C, clss

    distance = np.linalg.norm(X[:, None] - C, axis=-1)
    clss = np.argmin(distance, axis=-1)

    return C, clss

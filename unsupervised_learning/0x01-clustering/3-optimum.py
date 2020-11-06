#!/usr/bin/env python3
"""
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans                                  
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    tests for the optimum number of clusters by variance:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of
    clusters to check for (inclusive)
    - kmax: positive integer containing the maximum number of
    clusters to check for (inclusive)
    - iterations: positive integer containing the maximum number
    of iterations for K-means
    This function should analyze at least 2 different cluster sizes

    Returns: results, d_vars, or None, None on failure
        - results: list containing the outputs of K-means for
        each cluster size
        - d_vars: list containing the difference in variance
        from the smallest cluster size for each cluster size

    """

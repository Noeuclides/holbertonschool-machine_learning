#!/usr/bin/env python3
"""
Module to get expectection in EM
"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    calculates the expectation step in the EM algorithm for a GMM:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - pi: numpy.ndarray of shape (k,) containing the priors for
    each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster

    Returns: g, l, or None, None on failure
        - g: numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
        - l: total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    k = pi.shape[0]
    n, d = X.shape
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    g_sum = np.sum(g, axis=0)

    return g / g_sum, np.sum(np.log(g_sum))

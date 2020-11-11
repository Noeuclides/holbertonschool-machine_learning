#!/usr/bin/env python3
"""
Module to get Maximization step
"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - g: numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure
        - pi: numpy.ndarray of shape (k,) containing the updated
        priors for each cluster
        - m: numpy.ndarray of shape (k, d) containing the updated
        centroid means for each cluster
        - S: numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray):
        return None, None, None
    if not isinstance(g, np.ndarray):
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    if X.ndim != 2 or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), np.ones((n, ))).all():
        return None, None, None

    pi = np.zeros((k,))
    m = np.zeros((k, d))
    s = np.zeros((k, d, d))

    for i in range(k):
        pi[i] = np.sum(g[i]) / n
        m[i] = np.dot(g[i], X) / np.sum(g[i])
        X_m = X - m[i]
        s[i] = np.dot(g[i] * X_m.T, X_m) / np.sum(g[i])

    return pi, m, s

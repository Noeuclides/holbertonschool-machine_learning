#!/usr/bin/env python3
"""
Module to get EM
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    performs the expectation maximization for a GMM:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number of
    iterations for the algorithm
    - tol: non-negative float containing tolerance of the log
    likelihood, used to determine early stopping i.e. if the difference
    is less than or equal to tol you should stop the algorithm
    - verbose: boolean that determines if you should print information
    about the algorithm
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        - pi: numpy.ndarray of shape (k,) containing the priors for
        each cluster
        - m: numpy.ndarray of shape (k, d) containing the centroid
        means for each cluster
        - S: numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster
        - g: numpy.ndarray of shape (k, n) containing the probabilities
        for each data point in each cluster
        l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    i = 0
    l_prev = 0
    pi, mean, cov = initialize(X, k)
    g, log_like = expectation(X, pi, mean, cov)
    while i < iterations:
        if (np.abs(l_prev - log_like)) <= tol:
            break
        l_prev = log_like

        if verbose is True and (i % 10 == 0):
            rounded = log_like.round(5)
            print("Log Likelihood after {} iterations: {}".format(i, rounded))

        pi, mean, cov = maximization(X, g)
        g, log_like = expectation(X, pi, mean, cov)
        i += 1

    if verbose is True:
        rounded = log_like.round(5)
        print("Log Likelihood after {} iterations: {}".format(i, rounded))

    return pi, mean, cov, g, log_like

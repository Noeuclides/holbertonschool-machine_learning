#!/usr/bin/env python3
"""
Module to find clusters for a GMM
"""
import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    finds the best number of clusters for a GMM using the Bayesian
    Information Criterion:

    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing the minimum number of
    clusters to check for
    - kmax: positive integer containing the maximum number of
    clusters to check for
    - iterations: positive integer containing the maximum number of
    iterations for the EM algorithm
    - tol: non-negative float containing the tolerance for the EM algorithm
    - verbose: boolean that determines if the EM algorithm should print
    information to the
    standard output
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        - best_k: best value for k based on its BIC
        - best_result: tuple containing pi, m, S
            - pi: numpy.ndarray of shape (k,) containing the cluster priors
            for the best number of clusters
            - m: numpy.ndarray of shape (k, d) containing the centroid means
            for the best number of clusters
            - S: numpy.ndarray of shape (k, d, d) containing the covariance
            matrices for the best number of clusters
        - l: numpy.ndarray of shape (kmax - kmin + 1) containing the log
        likelihood for each cluster size tested
        - b is a numpy.ndarray of shape (kmax - kmin + 1) containing the
        BIC value for each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
            p: number of parameters required for the model
            n: number of data points used to create the model
            l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return (None, None, None, None)
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= X.shape[0]:
        return (None, None, None, None)
    if not isinstance(kmax, int) or kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin >= kmax:
        return (None, None, None, None)
    if not isinstance(iterations, int) or iterations <= 0:
        return (None, None, None, None)
    if not isinstance(tol, float) or tol <= 0:
        return (None, None, None, None)
    if not isinstance(verbose, bool):
        return None, None, None, None

    k_results = []
    results = []
    likehood_total = []
    bics = []
    n, d = X.shape
    for k in range(kmin, kmax + 1):
        pi, m, S, g, likehood_log = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        k_results.append(k)
        results.append((pi, m, S))
        likehood_total.append(likehood_log)
        p = (k * d * (d + 1) / 2) + (d * k) + k - 1
        bic = p * np.log(n) - 2 * likehood_log
        bics.append(bic)
    b = np.asarray(bics)
    best_b = np.argmin(b)
    l_total = np.asarray(likehood_total)

    return k_results[best_b], results[best_b], l_total, b

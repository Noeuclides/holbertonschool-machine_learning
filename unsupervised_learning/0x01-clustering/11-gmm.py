#!/usr/bin/env python3
"""
Module to get GMM with sklearn
"""
import sklearn.mixture


def gmm(X, k):
    """
    calculates a GMM from a dataset:

    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: number of clusters
    Returns: pi, m, S, clss, bic
        - pi: numpy.ndarray of shape (k,) containing the cluster priors
        - m: numpy.ndarray of shape (k, d) containing the centroid means
        - S: numpy.ndarray of shape (k, d, d) containing the
        covariance matrices
        - clss: numpy.ndarray of shape (n,) containing the cluster indices
        for each data point
        - bic: numpy.ndarray of shape (kmax - kmin + 1) containing the
        BIC value for each cluster size tested
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    return (
        gmm.weights_, gmm.means_, gmm.covariances_,
        gmm.predict(X), gmm.bic(X)
        )

#!/usr/bin/env python3
"""
Module to perform Principal component Analisys
"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset
    - X: numpy.ndarray of shape (n, d) where:
        - n: number of data points
        - d: number of dimensions in each point
    - ndim: new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the
    transformed version of X
    X = U ∑ V
    1. T = X W, V = W,
    2. T = U ∑
    """
    X_zero_mean = X - np.mean(X, axis=0)
    _, _, vh = np.linalg.svd(X_zero_mean)
    T = np.matmul(X_zero_mean, vh[:ndim].T)

    return T

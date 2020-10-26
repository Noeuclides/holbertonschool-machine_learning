#!/usr/bin/env python3
"""
Module to calculate the mean an covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """
    calculates the mean and covariance of a data set:

    - X: numpy.ndarray of shape (n, d) containing the data set:
        - n: number of data points
        - d: number of dimensions in each data point
    Returns: mean, cov:
        - mean: numpy.ndarray of shape (1, d) containing the data set mean
        - cov: numpy.ndarray of shape (d, d) containing the covariance matrix.

    > COV(x, y) = ((∑i=1->n) (xi - xmean)(yi - ymean)) / (n - 1)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0)
    data_mean = X - mean
    cov = (np.dot(data_mean.T, data_mean)) * (1 / X.shape[0])

    return np.array([mean]), cov

#!/usr/bin/env python3
"""
Module to calculate the correlation matrix
"""
import numpy as np


def correlation(C):
    """
    Write a function def correlation(C): that calculates a correlation matrix:

    - C: numpy.ndarray of shape (d, d) containing a covariance matrix
        - d: number of dimensions
    Returns a numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")

    r = np.zeros(shape=C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            den_coef1 = C[i][i] ** (1 / 2)
            den_coef2 = C[j][j] ** (1 / 2)
            r[i][j] = C[i][j] / (den_coef1 * den_coef2)

    return r

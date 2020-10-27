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

    rij = Cij / sqrt(Cii * Cjj)
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    diag = np.diag(C).reshape(1, C.shape[0])
    r = C / np.sqrt(diag.T * diag)

    return r

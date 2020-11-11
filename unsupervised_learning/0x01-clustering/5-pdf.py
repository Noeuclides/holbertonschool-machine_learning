#!/usr/bin/env python3
"""
Module to get the pdf
"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distrib::

    - X: numpy.ndarray of shape (n, d) containing the data points whose
    PDF should be evaluated
    - m: numpy.ndarray of shape (d,) containing the mean of the distrib.
    - S: numpy.ndarray of shape (d, d) containing the covariance of the
    distribution

    Returns: P, or None on failure
        - P: numpy.ndarray of shape (n,) containing the PDF values for
        each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None

    d = S.shape[0]
    det = np.linalg.det(S)
    den = np.sqrt(det) * ((2 * np.pi) ** (d / 2))
    inverse = np.linalg.inv(S)
    exponent = np.dot((X - m), inverse) * (X - m) / 2
    pdf = np.exp(np.sum(-exponent, axis=1)) / den

    return np.maximum(pdf, 1e-300)

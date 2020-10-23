#!/usr/bin/env python3
"""
check definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix:

    - matrix: numpy.ndarray of shape (n, n) whose definiteness
    should be calculated
    Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite
    if the matrix is positive definite, positive semi-definite,
    negative semi-definite, negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.array_equal(matrix.T, matrix):
        return None
    w, _ = np.linalg.eig(matrix)
    if all([item > 0 for item in w]):
        return "Positive definite"
    elif all([item < 0 for item in w]):
        return "Negative definite"
    elif any([item > 0 for item in w]) and w.min() == 0:
        return "Positive semi-definite"
    elif any([item < 0 for item in w]) and w.max() == 0:
        return "Negative semi-definite"
    elif any([item > 0 for item in w]) and any([item < 0 for item in w]):
        return "Indefinite"

    return None

#!/usr/bin/env python3
"""
module to perform Principal Component Analisys
"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset:

    - X: numpy.ndarray of shape (n, d) where:
        - n: number of data points
        - d: number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    - var: fraction of the variance that the PCA transformation should maintain
    Returns: the weights matrix, W, that maintains var fraction of X‘s
    original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
    of the transformed X

    """
    _, s, vh = np.linalg.svd(X)
    sv_sum = np.cumsum(s)
    filter = np.where(sv_sum < sv_sum[-1] * var)
    index = len(filter[0]) + 1

    return vh[:index].T

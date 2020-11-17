#!/usr/bin/env python3
"""
Absorbing chains module
"""
import numpy as np


def absorbing(P):
    """
    determines if a markov chain is absorbing:

    - P: square 2D numpy.ndarray of shape (n, n) representing the standard
    transition matrix
        - P[i, j]: probability of transitioning from state i to state j
        - n: number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if not ((P <= 1).all() and (P >= 0).all()):
        return False
    if (np.sum(P, axis=1) != 1).any():
        return False
    if (np.diag(P) == 1).all():
        return True
    if not (np.diag(P) == 1).any():
        return False

    n = P.shape[0]
    ones_idx = np.where(np.diag(P == 1))[0]

    for i in ones_idx:
        check = check_absorbing(P, P[:, ones_idx[i]], [i])
        if check:
            return True
    return False


def check_absorbing(P, column, index):
    if len(index) == P.shape[0]:
        return False
    if (np.delete(column, index) > 0).all():
        return True
    if (np.delete(column, index) == 0).all():
        return False

    return False

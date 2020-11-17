#!/usr/bin/env python3
"""
Regular chains module
"""
import numpy as np


def regular(P):
    """
    determines the steady state probabilities of a regular markov chain:

    - P: square 2D numpy.ndarray of shape (n, n) representing the
    transition matrix
        - P[i, j]: probability of transitioning from state i to state j
        - n: number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not ((P < 1).all() and (P > 0).all()):
        return None
    if (np.sum(P, axis=1) != 1).any():
        return None

    n = P.shape[0]
    s = P - np.eye(n)
    ones = np.ones((n, 1))
    s = np.concatenate([s, ones], axis=1)
    s = np.dot(s, s.T)

    return np.linalg.solve(s, ones).T

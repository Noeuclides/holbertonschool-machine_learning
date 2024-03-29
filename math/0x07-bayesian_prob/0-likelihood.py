#!/usr/bin/env python3
"""
Module to get the likelihood
"""
import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects:

    - x: number of patients that develop severe side effects
    - n: total number of patients observed
    - P: 1D numpy.ndarray containing the various hypothetical probabilities
    of developing severe side effects

    Returns: a 1D numpy.ndarray containing the likelihood of obtaining the
    data, x and n, for each probability in P, respectively
    Assume that x follows a binomial distribution.
    P(A|B) = P(B|A) * P(A) / P(B)
    likelihood = P(B|A)
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")

    fac = np.math.factorial
    combinatory = fac(n) / (fac(x) * fac(n - x))
    binomial_prob = combinatory * ((P ** x) * ((1 - P) ** (n - x)))

    return binomial_prob

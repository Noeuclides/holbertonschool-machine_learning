#!/usr/bin/env python3
"""
Module to get the posterior probability (bayes theorem)
"""
import numpy as np


def posterior(x, n, P, Pr):
    """
    calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data

    - x: number of patients that develop severe side effects
    - n: total number of patients observed
    - P: 1D numpy.ndarray containing the various hypothetical probabilities
    of developing severe side effects
    - Pr: 1D numpy.ndarray containing the prior beliefs of P

    Returns: a 1D numpy.ndarray containing the intersection of obtaining
    x and n with each probability in P, respectively
    Assuming that x follows a binomial distribution.
    P(A|B) = P(B|A) * P(A) / P(B)
    likelihood = P(B|A)
    intersection = P(BnA)
    marginal_prob = P(B)
    """
    if not isinstance(n, int) and n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x <= 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if any([False for i in P if i < 0 or i > 1]):
        raise ValueError("All values in P must be in the range [0, 1]")
    if any([False for i in Pr if i < 0 or i > 1]):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    fac = np.math.factorial
    combinatory = fac(n) / (fac(x) * fac(n - x))
    likelihood = combinatory * ((P ** x) * (1 - P) ** (n - x))
    intersection = likelihood * Pr
    marginal = np.sum(intersection)

    return intersection / marginal
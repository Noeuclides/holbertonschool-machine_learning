#!/usr/bin/env python3
"""
Module to get the posterior probability (bayes theorem)
"""
import numpy as np
from scipy import special


def posterior(x, n, p1, p2):
    """
    calculates the posterior probability that the probability of developing
    severe side effects falls within a specific range given the data:

    - x: number of patients that develop severe side effects
    - n: total number of patients observed
    - p1: lower bound on the range
    - p2: upper bound on the range
    Assume the prior beliefs of p follow a uniform distribution

    Returns: the posterior probability that p is within the range [p1, p2]
    given x and n
    """
    if not isinstance(n, int) and n < 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) and x <= 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    if x < p1:
        pdf_uniform = 0
    elif x >= p2:
        pdf_uniform = 1
    else:
        pdf_uniform = (x - p1) / (p2 - p1)

    g = special.gamma
    gamma = g(p1) * g(p2) / g(p1 + p2)
    likelihood = (x ** (p1 - 1)) * ((1 - x) ** (p2 - 1)) / gamma
    print(likelihood)
    intersection = likelihood * pdf_uniform
    print(intersection)
    marginal = np.sum(intersection)
    print(marginal)
    return intersection / marginal

#!/usr/bin/env python3
"""
module of Exponential distribution
"""


class Exponential:
    """
    class that represents a exponential distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """initialize class
        """
        self.lambtha = float(lambtha)
        if not data:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = 1 / (float(sum(data) / len(data)))

    def pdf(self, x):
        """probability distribution function
        """
        if x < 0:
            return 0

        lam = self.lambtha
        pdf = (self.e ** (-lam * x)) * lam
        return pdf

    def cdf(self, x):
        """cumulative distribution function
        """
        if x < 0:
            return 0
        cdf = 0
        lam = self.lambtha
        cdf = 1 - (self.e ** (-lam * x))
        return cdf

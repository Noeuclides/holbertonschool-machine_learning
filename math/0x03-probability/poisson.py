#!/usr/bin/env python3
"""
module of Poisson distribution
"""


class Poisson:
    """
    class that represents a poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """initialize class
        """
        self.lambtha = float(lambtha)
        if not data:
            if lambtha < 0:
                raise ValueError('lambtha must be a positive value')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """returns factorial of a number
        """
        if not isinstance(n, int):
            n = int(n)
        factorial = 1
        for i in range(n):
            factorial *= i + 1
        return factorial

    def pmf(self, k):
        """probability mass function
        """
        if k > 449:
            return 0

        lam = self.lambtha
        fack = self.factorial(k)
        pmf = (self.e ** (-lam) * lam ** k) / fack
        return pmf

    def cdf(self, k):
        """cumulative distribution function
        """
        if k > 449:
            return 0
        cdf = 0
        lam = self.lambtha
        for i in range(k + 1):
            fack = self.factorial(i)
            cdf += (self.e ** (-lam) * lam ** i) / fack
        return cdf
#!/usr/bin/env python3
"""
module of Normal distribution
"""


class Binomial:
    """
    class that represents a binomial distribution
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """initialize class
        """
        self.n = int(n)
        self.p = float(p)
        if not data:
            self.n = n
            self.p = p
            if n <= 0:
                raise ValueError('n must be a positive value')
            if not 0 < p < 1:
                raise ValueError('p must be greater than 0 and less than 1')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.n = n / 2
            self.p = p

    def pmf(self, k):
        """probability mass function
        """
        if not isinstance(k, int):
            k = int(k)

        if k > 499:
            return 0

        nfac = self.factorial(self.n)
        kfac = self.factorial(k)
        n_mfac = self.factorial(self.n - k)
        prb = (self.p ** k) * (1 - self.p) ** (self.n - k)
        pmf = nfac * prb / (kfac * n_mfac)
        return pmf

    def cdf(self, k):
        """cumulative distribution function
        """
        if not isinstance(k, int):
            k = int(k)
        if k > 499:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf

    def factorial(self, n):
        """returns factorial of a number
        """
        if not isinstance(n, int):
            n = int(n)
        factorial = 1
        for i in range(n):
            factorial *= i + 1
        return factorial

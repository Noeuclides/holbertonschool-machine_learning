#!/usr/bin/env python3
"""
module of Normal distribution
"""


class Normal:
    """
    class that represents a normal distribution
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """initialize class
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if not data:
            self.mean = mean
            self.stddev = stddev
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = (float(sum(data) / len(data)))
            sumdif = sum([(d - self.mean) ** 2 for d in data])
            self.stddev = (sumdif / len(data)) ** (1 / 2)

    def z_score(self, x):
        """z score calculation
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x value given z score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """probability density function
        """
        variance = self.stddev ** 2
        exp = -((x - self.mean) ** 2) / (2 * variance)
        den = (2 * self.pi * variance) ** (1 / 2)
        return (self.e ** exp) / den

    def cdf(self, x):
        """cumulative distribution function
        """
        error = self.erf((x - self.mean) / (self.stddev * (2 ** (1 / 2))))

        return ((1 + error) / 2)

    def erf(self, x):
        """error function
        """
        serie = 0
        for i in range(5):
            j = 2 * i + 1
            den = self.factorial(i) * j
            if j in [3, 7]:
                serie += -(x ** (j)) / den
            elif j in [1, 5, 9]:
                serie += (x ** (j)) / den
        return serie * 2 / (self.pi ** (1 / 2))

    def factorial(self, n):
        """returns factorial of a number
        """
        if not isinstance(n, int):
            n = int(n)
        factorial = 1
        for i in range(n):
            factorial *= i + 1
        return factorial

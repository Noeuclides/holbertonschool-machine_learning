#!/usr/bin/env python3
"""
module to summarize
"""


def summation_i_squared(n):
    """
    calculate sum
    """

    if n is None or n < 1:
        return None
    if not isinstance(n, (int, float)):
        return None
    return int(sum(map(lambda x: x ** 2, range(1, (n + 1)))))

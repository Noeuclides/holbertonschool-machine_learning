#!/usr/bin/env python3
"""module to moving average
"""

import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    - data: list of data to calculate the moving average of
    - beta: weight used for the moving average
    use bias correction
    Returns: a list containing the moving averages of data
    """
    moving_avg = []
    v = 0
    for t, element in enumerate(data):
        bias_correction = 1 - beta ** (t + 1)
        v = v * beta + (1 - beta) * element
        moving_avg.append(v / bias_correction)

    return moving_avg

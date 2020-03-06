#!/usr/bin/env python3
"""module to moving average
"""

import tensorflow as tf
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set
    """
    mov_avg = []
    v = 0
    for i in range(1, len(data) + 1):
        bias_avg = 1 - beta ** i
        v = v * beta + (1 - beta) * data[i - 1]
        mov_avg.append(v / bias_avg)

    return mov_avg

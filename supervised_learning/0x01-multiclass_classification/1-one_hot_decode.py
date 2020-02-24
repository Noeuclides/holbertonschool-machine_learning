#!/usr/bin/env python3
"""module with a on hot encode function
"""

import numpy as np


def one_hot_decode(one_hot):
    """convert a one-hot matrix into a vector of labels
    """
    if one_hot.size == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not isinstance(one_hot, np.ndarray):
        return None
    if one_hot.max() != 1 or one_hot.min() != 0:
        return None
    decode = np.zeros((one_hot.shape[1]), dtype=int)
    for i in range(len(one_hot)):
        p = np.where(one_hot[i] == 1)
        decode[p] = i
    return decode

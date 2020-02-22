#!/usr/bin/env python3
"""module with a on hot encode function
"""

import numpy as np


def one_hot_decode(one_hot):
    """convert a one-hot matrix into a vector of labels
    """
    decode = np.zeros((one_hot.shape[1]), dtype=int)
    p = np.where(one_hot == 1)
    for i in range(len(one_hot)):
        p = np.where(one_hot[i] == 1)
        decode[p] = i
    return decode

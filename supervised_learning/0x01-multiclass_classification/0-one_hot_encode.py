#!/usr/bin/env python3
"""module with a on hot encode function
"""

import numpy as np


def one_hot_encode(Y, classes):
    """convert a numeric label vector into a one-hot matrix
    """
    if classes < Y.shape[0]:
        return None
    encode = np.zeros((classes, Y.shape[0]))
    rows = np.arange(Y.shape[0])
    encode[Y, rows] = 1
    return encode

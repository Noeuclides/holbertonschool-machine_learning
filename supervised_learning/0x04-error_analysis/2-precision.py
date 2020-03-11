#!/usr/bin/env python3
"""precision module
"""

import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix
    """
    true_pos = np.diagonal(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    precision = true_pos / (false_pos + true_pos)

    return precision

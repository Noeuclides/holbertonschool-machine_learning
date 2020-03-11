#!/usr/bin/env python3
"""module of sensitivity confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix
    """

    true_pos = np.diagonal(confusion)
    sensitivity = true_pos / np.sum(confusion, axis=1)

    return sensitivity

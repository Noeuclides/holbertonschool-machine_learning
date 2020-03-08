#!/usr/bin/env python3
"""module of sensitivity confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix
    """

    sensitivity = []
    for i in range(confusion.shape[0]):
        true_pos = confusion[i][i]
        false_neg = sum(confusion[i]) - true_pos

        sensitivity.append(true_pos / (true_pos + false_neg))
    return sensitivity

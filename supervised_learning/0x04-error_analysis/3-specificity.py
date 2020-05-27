#!/usr/bin/env python3
"""specificity module
"""

import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    - confusion: numpy.ndarray of shape (classes, classes) where row
    indices represent the correct labels and column indices represent
    the predicted labels
        - classes: number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the
    specificity of each class
    """
    true_pos = np.diagonal(confusion)
    false_pos = np.sum(confusion, axis=0) - true_pos
    false_neg = np.sum(confusion, axis=1) - true_pos
    true_neg = np.sum(confusion) - (true_pos + false_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)

    return specificity

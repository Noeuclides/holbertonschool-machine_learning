#!/usr/bin/env python3
"""module of sensitivity confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    - confusion: numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the predicted labels
        - classes: number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of each class
    """
    true_positive = confusion.diagonal()
    sensitivity = true_positive / np.sum(confusion, axis=1)

    return sensitivity

#!/usr/bin/env python3
"""precision module
"""

import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix
    - confusion: numpy.ndarray of shape (classes, classes) where row indices
    represent the correct labels and column indices represent the predicted
    labels
        - classes: number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the precision
    of each class
    """
    true_positive = confusion.diagonal()
    logits = np.sum(confusion, axis=0)
    precision = true_positive / logits

    return precision

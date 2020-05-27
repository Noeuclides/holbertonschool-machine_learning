#!/usr/bin/env python3
"""f1 score
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
    - confusion: numpy.ndarray of shape (classes, classes) where row
    indices represent the correct labels and column indices represent
    the predicted labels.
    - classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) containing the F1 score
    of each class
    """
    sens = sensitivity(confusion)
    prcs = precision(confusion)
    f1_score = 2 * sens * prcs / (sens + prcs)

    return f1_score

#!/usr/bin/env python3
"""module confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    - labels: one-hot numpy.ndarray of shape (m, classes) containing
    the correct labels for each data point
        - m: number of data points
        - classes: number of classes
    - logits: one-hot numpy.ndarray of shape (m, classes) containing
    the predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes) with
    row indices representing the correct labels and column indices representing
    the predicted labels
    """
    return np.matmul(labels.T, logits)

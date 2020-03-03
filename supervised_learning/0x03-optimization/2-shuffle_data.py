#!/usr/bin/env python3
"""Sheffle data
"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way
    """
    return np.random.permutation(X), np.random.permutation(Y)

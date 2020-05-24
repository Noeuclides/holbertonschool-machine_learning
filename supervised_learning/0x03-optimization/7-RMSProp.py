#!/usr/bin/env python3
"""module RMSprop
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    updates a variable using the RMSProp optimization algorithm
    - alpha: learning rate
    - beta2: RMSProp weight
    - epsilon: small number to avoid division by zero
    - var is a numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - s: previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    sqrt = s ** (1 / 2) + epsilon
    var = var - alpha * grad / sqrt

    return var, s

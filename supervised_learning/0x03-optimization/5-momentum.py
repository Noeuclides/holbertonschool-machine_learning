#!/usr/bin/env python3
"""Module of momentum gradient descent
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent
    with momentum optimization algorithm
    - alpha: learning rate
    - beta1: momentum weight
    - var: numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - v: previous first moment of var
    Returns: the updated variable and the new moment, respectively
    Vdw = ß * Vdw + (1 - ß) * dW
    W = W - alpha * Vdw
    """
    v = v * beta1 + (1 - beta1) * grad
    var = var - alpha * v

    return var, v

#!/usr/bin/env python3
"""module RMSprop
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm
    """
    rms = grad * grad
    s = beta2 * s + (1 - beta2) * rms
    den = s ** (1 / 2) + epsilon
    var = var - alpha * grad / den

    return var, s

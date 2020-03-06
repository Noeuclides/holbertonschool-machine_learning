#!/usr/bin/env python3
"""module to Adam
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm
    """
    v = beta1 * v + (1 - beta1) * grad
    rms = grad * grad
    s = beta2 * s + (1 - beta2) * rms

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    den = s_corrected ** (1 / 2) + epsilon
    var = var - alpha * (v_corrected / den)

    return var, v_corrected, s_corrected

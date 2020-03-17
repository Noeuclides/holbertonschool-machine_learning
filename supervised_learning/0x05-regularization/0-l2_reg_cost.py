#!/usr/bin/env python3
"""L2 Regularization Cost
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    calculates the cost of a NN with L2 regularization
    - cost: cost of the network without L2 regularization
    - lambtha: regularization parameter
    - weights: dictionary of the weights and biases (numpy.ndarrays) of the NN
    - L: number of layers in the NN
    - m is the number of data points used
    """
    for i in range(L):
        key = "W{}".format(i + 1)
        l2_reg = np.linalg.norm(weights[key])
        cost = cost + lambtha * l2_reg / (2 * m)

    return cost

#!/usr/bin/env python3
"""L2 Regularization Cost
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:
    - cost: tensor containing the cost of the network without L2 regularization
    Returns: a tensor containing the cost of the network accounting
    for L2 regularization
    """
    l2_reg_cost = cost + tf.contrib.layers.l2_regularizer(scale=0.1)

    return l2_reg_cost

#!/usr/bin/env python3
"""module rmsprop ugraded
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm
    """
    optimizer = tf.train.RMSPropOptimizer(alpha,
                                          decay=beta2,
                                          epsilon=epsilon)
    rmsprop = optimizer.minimize(loss)

    return rmsprop

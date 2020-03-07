#!/usr/bin/env python3
"""module adam ugraded
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tensorflow using the adam optimization algorithm
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta 1,
                                       beta2, epsilon)
    adam = optimizer.minimize(loss)

    return adam

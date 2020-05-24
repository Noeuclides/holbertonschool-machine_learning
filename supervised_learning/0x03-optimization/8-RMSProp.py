#!/usr/bin/env python3
"""module rmsprop ugraded
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm
    - loss: loss of the network
    - alpha: learning rate
    - beta2: RMSProp weight
    - epsilon: small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    # Construct a new RMSProp optimizer
    rmsprop = tf.train.RMSPropOptimizer(alpha,
                                        decay=beta2,
                                        epsilon=epsilon)
    # minimize combines calls compute_gradients() and apply_gradients()
    minimize = rmsprop.minimize(loss)

    return minimize

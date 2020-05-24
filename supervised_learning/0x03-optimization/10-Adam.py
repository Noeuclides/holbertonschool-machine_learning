#!/usr/bin/env python3
"""module adam ugraded
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon): 
    """
    creates the training operation for a neural network
    in tensorflow using the adam optimization algorithm
    - loss: loss of the network
    - alpha: learning rate
    - beta1: weight used for the first moment
    - beta2: weight used for the second moment
    - epsilon: small number to avoid division by zero
    Returns: the Adam optimization operation
    """
    adam = tf.train.AdamOptimizer(alpha, beta1,
                                  beta2, epsilon)
    # combines calls compute_gradients() and apply_gradients()
    minimize = adam.minimize(loss)

    return minimize

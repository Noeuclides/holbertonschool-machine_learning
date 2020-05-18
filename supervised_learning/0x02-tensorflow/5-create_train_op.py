#!/usr/bin/env python3
"""Module to create a train operation
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    creates the training operation for the network
    - loss: loss of the network’s prediction
    - alpha:learning rate
    Returns: an operation that trains the network using gradient descent

    """
    optimize = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimize.minimize(loss)

    return train_op

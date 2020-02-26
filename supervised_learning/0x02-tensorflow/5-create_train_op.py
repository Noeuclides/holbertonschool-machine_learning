#!/usr/bin/env python3
"""
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """create the training operation for the network
    """
    optimize = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimize.minimize(loss)

    return train_op

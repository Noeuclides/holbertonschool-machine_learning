#!/usr/bin/env python3
"""module momentum ugraded
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates the training operation for a neural network
    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return momentum

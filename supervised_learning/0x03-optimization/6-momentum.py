#!/usr/bin/env python3
"""module momentum ugraded
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm:    
    - loss: loss of the network
    - alpha: learning rate
    - beta1_ momentum weight
    Returns: the momentum optimization operation
    """
    momentum = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)

    return momentum

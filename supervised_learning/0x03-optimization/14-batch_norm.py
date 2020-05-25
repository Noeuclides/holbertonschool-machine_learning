#!/usr/bin/env python3
"""
modulo to Batch Normalization with tf
"""

import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:
    - prev: activated output of the previous layer
    - n: number of nodes in the layer to be created
    - activation: activation function used on the output of the layer
    tf.layers.Dense layer is the base layer with kernal initializer
    tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    the layer incorporates two trainable parameters, gamma and beta,
    initialized as vectors of 1 and 0 respectively
    uses an epsilon of 1e-8
    Returns: a tensor of the activated output for the layer
    """
    # activation is done after the batch normalization
    weights_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=weights_init)
    Z = layer(prev)

    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))

    mean, variance = tf.nn.moments(Z, axes=0)
    batch_norm = tf.nn.batch_normalization(
        Z, mean, variance, offset=beta, scale=gamma, variance_epsilon=1e-8)

    return activation(batch_norm)

#!/usr/bin/env python3
"""module to get the tensor output of a layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    - prev: tensor output of the previous layer
    - n: number of nodes in the layer to create
    - activation: activation function that the layer should use
    Returns: the tensor output of the layer
    """
    # tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    # implements He et. al initialization for the layer weights
    weights_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            name='layer',
                            kernel_initializer=weights_init)

    return layer(prev)

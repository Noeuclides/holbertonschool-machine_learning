#!/usr/bin/env python3
"""module to get the tensor output of a layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """the tensor output of the layer
    """
    weights_init = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            name='layer',
                            kernel_initializer=weights_init)

    return layer(prev)

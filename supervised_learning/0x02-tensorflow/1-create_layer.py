#!/usr/bin/env python3
"""module to get the tensor output of a layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """the tensor output of the layer
    """
    layer = tf.layers.dense(prev, units=n, activation=activation, name='layer')
    tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    return layer

#!/usr/bin/env python3
"""module to placeholders NN
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """eturns two placeholders, x and y, for the neural network
    """
    x = tf.placeholder('float32', [None, nx], name='x')
    y = tf.placeholder('float32', [None, classes], name='y')

    return x, y

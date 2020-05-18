#!/usr/bin/env python3
"""module to placeholders NN
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    returns two placeholders, x and y, for the neural network
    - nx: number of feature columns in our data
    - classes: number of classes in our classifier
    Returns: placeholders named x and y, respectively
    - x: placeholder for the input data to the neural network
    - y: placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder('float32', [None, nx], name='x')
    y = tf.placeholder('float32', [None, classes], name='y')

    return x, y

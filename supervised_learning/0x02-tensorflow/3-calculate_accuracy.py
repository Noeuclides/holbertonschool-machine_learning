#!/usr/bin/env python3
"""accuracy NN
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    calculates the accuracy of a prediction
    - y: placeholder for the labels of the input data
    - y_pred: tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    compare = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    cast = tf.cast(compare, tf.float32)
    accuracy = tf.reduce_mean(cast)
    return accuracy

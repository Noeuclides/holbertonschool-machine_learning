#!/usr/bin/env python3
"""accuracy NN
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculate the accuracy of a prediction
    """
    compare = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y, axis=-1))
    cast = tf.cast(compare, tf.float32)
    accuracy = tf.reduce_mean(cast)
    return accuracy

#!/usr/bin/env python3
"""loss NN
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculate the softmax cross-entropy loss of a prediction
    """
    loss = tf.losses.softmax_cross_entropy(
        y, y_pred, reduction=tf.losses.Reduction.MEAN)

    return loss

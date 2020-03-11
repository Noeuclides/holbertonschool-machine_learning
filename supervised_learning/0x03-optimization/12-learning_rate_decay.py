#!/usr/bin/env python3
"""module learning decay ugraded
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates the training operation for a neural network
    in tensorflow using the learning decay optimization algorithm
    """
    optimizer = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)
    decay = optimizer.minimize(loss)

    return decay

#!/usr/bin/env python3
"""module learning decay ugraded
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tf using inverse time decay
    - alpha: original learning rate
    - decay_rate: weight used to determine the rate at which alpha will decay
    - global_step: number of passes of gradient descent that have elapsed
    - decay_step: number of passes of gradient descent that occurs before
    alpha is decayed further
    the learning rate decay occurs in a stepwise fashion
    Returns: the learning rate decay operation
    """
    learning_rate_decay = tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True)

    return learning_rate_decay

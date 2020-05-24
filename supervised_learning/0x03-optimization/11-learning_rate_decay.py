#!/usr/bin/env python3
"""module learning rate dacay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay in numpy
    - alpha: original learning rate
    - decay_rate: weight used to determine the rate at which alpha will decay
    - global_step: number of passes of gradient descent that have elapsed
    - decay_step: number of passes of gradient descent
    that should occur before alpha is decayed further

    learning rate decay occur in a stepwise fashion
    Returns: the updated value for alpha
    """
    global_step = int(global_step / decay_step)
    alpha = alpha / (1 + decay_rate * global_step)

    return alpha

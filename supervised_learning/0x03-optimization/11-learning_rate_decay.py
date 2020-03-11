#!/usr/bin/env python3
"""module learning rate dacay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate using inverse time decay in numpy
    """
    step = int(global_step / decay_step)
    den = 1 + decay_rate * step
    alpha = alpha / den

    return alpha

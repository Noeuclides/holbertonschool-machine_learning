#!/usr/bin/env python3
"""Optiize module
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    sets up Adam optimization for a keras model:
    - network: model to optimize
    - alpha: learning rate
    - beta1: first Adam optimization parameter
    - beta2: second Adam optimization parameter
    """
    K.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)

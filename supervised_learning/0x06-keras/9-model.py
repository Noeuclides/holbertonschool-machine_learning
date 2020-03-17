#!/usr/bin/env python3
"""save and load module
"""

import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model:
    """
    network.save(filename)

    return None


def load_model(filename):
    """loads an entire model
    """
    model = K.models.load_model(filename)

    return model

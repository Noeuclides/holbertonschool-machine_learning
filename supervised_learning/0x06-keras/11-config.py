#!/usr/bin/env python3
"""save and load weights
"""

import tensorflow.keras as K


def save_config(network, filename):
    """saves an entire model:
    """
    with open(filename, "w") as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """loads an entire model
    """
    with open(filename, "r") as f:
        model_load = f.read()

    model = K.models.model_from_json(model_load)

    return model

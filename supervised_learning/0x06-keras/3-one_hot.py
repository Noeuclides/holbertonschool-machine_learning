#!/usr/bin/env python3
"""one hot matrix module
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts a label vector into a one-hot matrix:
    """
    encode = K.utils.to_categorical(labels)

    return encode

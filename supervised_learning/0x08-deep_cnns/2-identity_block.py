#!/usr/bin/env python3
"""
Identity block base on paper:
Deep Residual Leaning for Image Recognition (2015)
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in
    Deep Residual Learning for Image Recognition (2015):

    - A_prev: output from the previous layer
    - filters: tuple or list containing F11, F3, F12, respectively:
        - F11: number of filters in the first 1x1 convolution
        - F3: number of filters in the 3x3 convolution
        - F12: number of filters in the second 1x1 convolution
    - All convolutions inside the block should be followed by
    batch normalization along the channels axis and a rectified
    linear activation (ReLU), respectively.
    - All weights should use he normal initialization
    - Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=None)

    conv_F11 = K.layers.Conv2D(F11, kernel_size=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer=initializer)(A_prev)
    batch_1 = K.layers.BatchNormalization(axis=3)(conv_F11)
    activation_1 = K.layers.Activation('relu')(batch_1)

    conv_F3 = K.layers.Conv2D(F3, kernel_size=(3, 3),
                              padding='same', activation='relu',
                              kernel_initializer=initializer)(activation_1)
    batch_2 = K.layers.BatchNormalization(axis=3)(conv_F3)
    activation_2 = K.layers.Activation('relu')(batch_2)

    conv_F12 = K.layers.Conv2D(F12, kernel_size=(1, 1),
                               padding='same', activation='relu',
                               kernel_initializer=initializer)(activation_2)
    batch_3 = K.layers.BatchNormalization(axis=3)(conv_F12)

    add_1 = K.layers.add([batch_3, A_prev])
    identity = K.layers.Activation('relu')(add_1)

    return identity

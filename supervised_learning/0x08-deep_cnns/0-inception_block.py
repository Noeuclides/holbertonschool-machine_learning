#!/usr/bin/env python3
"""
Inception Block based on paper:
Going Deeper with Convolutions(2014)
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in
    Going Deeper with Convolutions(2014):

    - A_prev: output from the previous layer
    - filters: tuple or list containing F1, F3R, F3,F5R, F5, FPP:
        - F1: number of filters in the 1x1 convolution
        - F3R: number of filters in the 1x1 convolution before the 3x3 conv.
        - F3: number of filters in the 3x3 convolution
        - F5R: number of filters in the 1x1 convolution before the 5x5 conv.
        - F5: number of filters in the 5x5 convolution
        - FPP: number of filters in the 1x1 convolution after the max pooling
    All convolutions should use a rectified linear activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    initializer = K.initializers.he_normal(seed=None)

    conv_F1 = K.layers.Conv2D(
        F1, kernel_size=(1, 1), padding='same',
        activation='relu', kernel_initializer=initializer)(A_prev)

    conv_F3R = K.layers.Conv2D(
        F3R, kernel_size=(1, 1), padding='same',
        activation='relu', kernel_initializer=initializer)(A_prev)
    conv_F3 = K.layers.Conv2D(
        F3, kernel_size=(3, 3), padding='same',
        activation='relu', kernel_initializer=initializer)(conv_F3R)

    conv_F5R = K.layers.Conv2D(
        F5R, kernel_size=(1, 1), padding='same',
        activation='relu', kernel_initializer=initializer)(A_prev)
    conv_F5 = K.layers.Conv2D(
        F5, kernel_size=(5, 5), padding='same',
        activation='relu', kernel_initializer=initializer)(conv_F5R)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=1, padding='same')(A_prev)
    conv_FPP = K.layers.Conv2D(
        FPP, kernel_size=(1, 1), padding='same',
        activation='relu', kernel_initializer=initializer)(max_pool)

    filter_concat = K.layers.concatenate([conv_F1, conv_F3, conv_F5, conv_FPP])

    return filter_concat

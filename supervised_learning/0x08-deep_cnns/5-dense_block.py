#!/usr/bin/env python3
"""
Dense block base on paper:
Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in
    Densely Connected Convolutional Networks:
    - X: output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - growth_rate: growth rate for the dense block
    - layers: number of layers in the dense block
    - bottleneck layers used for DenseNet-B
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the
    Dense Block and the number of filters within the concatenated outputs
    """
    initializer = K.initializers.he_normal(seed=None)

    for i in range(layers):
        batch1 = K.layers.BatchNormalization()(X)
        activation1 = K.layers.Activation('relu')(batch1)
        conv1 = K.layers.Conv2D(4 * growth_rate, kernel_size=(1, 1),
                                padding='same',
                                kernel_initializer=initializer)(activation1)

        batch2 = K.layers.BatchNormalization()(conv1)
        activation2 = K.layers.Activation('relu')(batch2)
        conv2 = K.layers.Conv2D(growth_rate, kernel_size=(3, 3),
                                padding='same',
                                kernel_initializer=initializer)(activation2)
        nb_filters += growth_rate

        X = K.layers.concatenate([X, conv2])

    return X, nb_filters

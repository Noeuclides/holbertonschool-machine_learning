#!/usr/bin/env python3
"""
DenseNet base on paper:
Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in
    Densely Connected Convolutional Networks:
    - X: output from the previous layer
    - nb_filters: integer representing the number of filters in X
    - compression: compression factor for the transition layer

    implement compression as used in DenseNet-C
    - All weights should use he normal initialization
    - All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    Returns: The output of the transition layer and the number of
    filters within the output, respectively
    """
    initializer = K.initializers.he_normal(seed=None)
    batch1 = K.layers.BatchNormalization()(X)
    activation1 = K.layers.Activation('relu')(batch1)
    filters = int(nb_filters * compression)
    conv1 = K.layers.Conv2D(filters, kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=initializer)(activation1)

    avg_pool1 = K.layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=2,
                                          padding='valid')(conv1)

    return avg_pool1, filters

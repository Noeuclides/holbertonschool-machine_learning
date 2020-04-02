#!/usr/bin/env python3
"""
DenseNet121 base on paper:
Densely Connected Convolutional Networks
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in
    Densely Connected Convolutional Networks:
    - growth_rate: growth rate
    - compression: compression factor
    - input data have shape (224, 224, 3)
    - All convolutions should be preceded by Batch Normalization
    and a rectified linear activation (ReLU), respectively
    - All weights should use he normal initialization
    - Returns: the keras model
    """
    X_input = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(seed=None)

    batch1 = K.layers.BatchNormalization()(X_input)
    activation1 = K.layers.Activation('relu')(batch1)
    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=2,
                            padding='same',
                            kernel_initializer=initializer)(activation1)
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=2, padding='same')(conv1)
    X, nb_filters = dense_block(max_pool, 64, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)
    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=None,
                                         padding='same')(X)

    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=initializer)(avg_pool)
    model = K.models.Model(inputs=X_input, outputs=dense)

    return model

#!/usr/bin/env python3
"""LeNet-5 wit Keras
"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras
    """
    kernel_init = K.initializers.he_normal()
    activation = 'relu'

    conv_layer = K.layers.Conv2D(6, (5, 5), strides=(1, 1),
                                 padding='same',
                                 activation=activation,
                                 kernel_initializer=kernel_init)(X)
    pooling = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(conv_layer)
    conv_layer = K.layers.Conv2D(16, (5, 5), strides=(1, 1),
                                 padding='valid', activation=activation,
                                 kernel_initializer=kernel_init)(pooling)
    pooling = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(conv_layer)

    flat_layer = K.layers.Flatten()(pooling)
    dense_layer = K.layers.Dense(120, activation=activation,
                                 kernel_initializer=kernel_init)(flat_layer)
    dense_layer = K.layers.Dense(84, activation=activation,
                                 kernel_initializer=kernel_init)(dense_layer)
    output = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=kernel_init)(dense_layer)

    model = K.models.Model(inputs=X, outputs=output)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model

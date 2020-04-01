#!/usr/bin/env python3
"""
ResNet-5 base on paper
Deep Residual Learning for Image Recognition (2015)
"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    builds the ResNet-50 architecture as described in
    Deep Residual Learning for Image Recognition (2015):
    - You can assume the input data will have shape (224, 224, 3)
    - All convolutions inside and outside the blocks should be followed
    by batch normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    - All weights should use he normal initialization
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(seed=None)

    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=2,
                            padding='same',
                            kernel_initializer=initializer)(X)
    batch1 = K.layers.BatchNormalization()(conv1)
    activation1 = K.layers.Activation('relu')(batch1)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                     padding='same')(activation1)

    projection = projection_block(max_pool, [64, 64, 256], s=1)
    identity = projection
    for i in range(2):
        identity = identity_block(identity, [64, 64, 256])

    projection = projection_block(identity, [128, 128, 512], s=2)
    identity = projection
    for i in range(3):
        identity = identity_block(identity, [128, 128, 512])

    projection = projection_block(identity, [256, 256, 1024], s=2)
    identity = projection
    for i in range(5):
        identity = identity_block(identity, [256, 256, 1024])

    projection = projection_block(identity, [512, 512, 2048], s=2)
    identity = projection
    for i in range(2):
        identity = identity_block(identity, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=1,
                                         padding='same')(identity)

    dense = K.layers.Dense(1000, activation='relu',
                           kernel_initializer=initializer)(avg_pool)

    model = K.models.Model(inputs=X, outputs=dense)

    return model

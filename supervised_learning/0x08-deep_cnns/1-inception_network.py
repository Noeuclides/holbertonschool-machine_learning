#!/usr/bin/env python3
"""
Inception based on paper
Going Deeper with Convolutions (2014)
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in
    Going Deeper with Convolutions (2014)
    - input data will have shape (224, 224, 3)
    - All convolutions inside and outside the inception block
    should use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    initializer = K.initializers.he_normal(seed=None)

    input_1 = K.Input(shape=(224, 224, 3))

    conv_1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=2,
                             padding='same', activation='relu',
                             kernel_initializer=initializer)(input_1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2, padding='same')(conv_1)

    conv_2 = K.layers.Conv2D(64, kernel_size=(1, 1), strides=1,
                             padding='same', activation='relu',
                             kernel_initializer=initializer)(max_pool1)
    conv_2depth = K.layers.Conv2D(192, kernel_size=(3, 3), strides=1,
                                  padding='same', activation='relu',
                                  kernel_initializer=initializer)(conv_2)

    max_pool2 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2, padding='same')(conv_2depth)

    inception_3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])
    inception_3b = inception_block(inception_3a, [128, 128, 192, 32, 96, 64])
    max_pool3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2, padding='same')(inception_3b)

    inception_4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])
    inception_4b = inception_block(inception_4a, [160, 112, 224, 24, 64, 64])
    inception_4c = inception_block(inception_4b, [128, 128, 256, 24, 64, 64])
    inception_4d = inception_block(inception_4c, [112, 144, 288, 32, 64, 64])
    inception_4e = inception_block(inception_4d, [256, 160, 320, 32, 128, 128])
    max_pool4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2, padding='same')(inception_4e)
    inception_5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])
    inception_5b = inception_block(inception_5a, [384, 192, 384, 48, 128, 128])
    avg_pool1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                          strides=1,
                                          padding='valid')(inception_5b)

    dropout = K.layers.Dropout(0.4)(avg_pool1)
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_initializer=initializer)(dropout)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model

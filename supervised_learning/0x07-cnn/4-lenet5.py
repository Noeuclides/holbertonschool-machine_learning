#!/usr/bin/env python3
"""
lenet5
"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu

    # convolution section
    conv_layer1 = tf.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        kernel_initializer=initializer,
        activation=activation)(x)
    max_pool = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                      strides=(2, 2))(conv_layer1)
    conv_layer2 = tf.layers.Conv2D(
        filters=16, kernel_size=(5, 5), padding='valid',
        kernel_initializer=initializer,
        activation=activation)(max_pool)
    max_pool = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                      strides=(2, 2))(conv_layer2)

    # Fully connected section
    flatten = tf.layers.Flatten()(max_pool)
    layer = tf.layers.Dense(units=120, activation=activation,
                            kernel_initializer=initializer)(flatten)
    layer = tf.layers.Dense(units=84, activation=activation,
                            kernel_initializer=initializer)(layer)
    layer = tf.layers.Dense(units=10,
                            kernel_initializer=initializer)(layer)

    # prediction performance
    output = tf.nn.softmax(layer)
    loss = tf.losses.softmax_cross_entropy(y, layer)
    train = tf.train.AdamOptimizer().minimize(loss)

    # metrics
    label = tf.argmax(y, 1)
    prediction = tf.argmax(layer, 1)
    compare = tf.equal(prediction, label)
    accuracy = tf.reduce_mean(tf.cast(compare, tf.float32))

    return (output, train, loss, accuracy)

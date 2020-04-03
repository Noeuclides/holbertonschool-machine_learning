#!/usr/bin/env python3
"""
lenet5
"""
import numpy as np
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()

    conv_layer1 = tf.layers.Conv2D(
        filters=6, kernel_size=(
            5, 5), padding='same')(x)
    max_pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv_layer1)
    conv_layer2 = tf.layers.Conv2D(
        filters=16, kernel_size=(
            5, 5), padding='valid')(max_pool)
    max_pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv_layer2)
    flatten = tf.layers.Flatten()(max_pool)
    layer = tf.layers.Dense(units=120, activation='relu')(flatten)
    layer = tf.layers.Dense(units=84, activation='relu')(layer)
    layer = tf.layers.Dense(units=10, activation='relu')(layer)
    layer = tf.nn.softmax(layer)
    loss = tf.losses.softmax_cross_entropy(y, layer)
    train = tf.train.AdamOptimizer().minimize(loss)

    val = tf.argmax(y, 1)
    pred = tf.argmax(layer, 1)
    eq = tf.equal(pred, val)
    acc = tf.reduce_mean(tf.cast(eq, tf.float32))

    return (layer, train, loss, acc)

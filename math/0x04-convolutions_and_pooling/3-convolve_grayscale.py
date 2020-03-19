#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def zero_pad(array, pad1, pad2):
    """
    padding an array with zeros
    """
    array_pad = np.pad(array, ((0, 0), (pad1, pad1), (pad2, pad2)),
                       'constant', constant_values=(0, 0))

    return array_pad


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a valid convolution on grayscale images:
    - images: numpy.ndarray, shape (m, h, w), with multiple grayscale images
    - kernel: numpy.ndarray, shape (kh, kw), with the kernel for the conv.
    - padding: tuple of (ph, pw),‘same’, or ‘valid’
    - stride: tuple of (sh, sw)
    """
    step_h = kernel.shape[0]
    step_w = kernel.shape[1]
    if isinstance(padding, tuple):
        images_pad = zero_pad(images, padding[0], padding[1])
    elif padding == 'valid':
        images_pad = zero_pad(images, int(step_h / 2), int(step_w / 2))
    else:
        images_pad = images
    window_h = images_pad.shape[1] - kernel.shape[0]
    window_w = images_pad.shape[2] - kernel.shape[1]
    m = images.shape[0]
    img = np.arange(m)
    convolution = np.zeros((m,
                            int(window_h / stride[0] + 1),
                            int(window_w / stride[1] + 1)))
    for i in range(0, int(window_h / stride[0] + 1)):
        for j in range(0, int(window_w / stride[1] + 1)):
            print(i, j)
            convole = images_pad[img, i:i + step_h, j:j + step_w] * kernel
            pos = np.sum(convole, axis=(1, 2))
            convolution[img, i, j] = pos

    return convolution

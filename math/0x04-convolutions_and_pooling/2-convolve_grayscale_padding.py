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


def convolve_grayscale_padding(images, kernel, padding): 
    """
    performs a valid convolution on grayscale images:
    - images: numpy.ndarray, shape (m, h, w), with multiple grayscale images
    - kernel: numpy.ndarray, shape (kh, kw), with the kernel for the conv.
    - padding: tuple of (ph, pw)
    """
    step_h = kernel.shape[0]
    step_w = kernel.shape[1]
    images_pad = zero_pad(images, padding[0], padding[1])
    window_h = images_pad.shape[1] - kernel.shape[0] + 1
    window_w = images_pad.shape[2] - kernel.shape[1] + 1
    m = images.shape[0]
    img = np.arange(m)

    convolution = np.zeros((m, window_h, window_w))
    for i in range(window_h):
        for j in range(window_w):
            convole = images_pad[img, i:i + step_h, j:j + step_w] * kernel
            pos = np.sum(convole, axis=(1, 2))
            convolution[img, i, j] = pos

    return convolution

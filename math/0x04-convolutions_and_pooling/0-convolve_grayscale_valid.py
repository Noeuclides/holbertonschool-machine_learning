#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images:
    - images: numpy.ndarray, shape (m, h, w), with multiple grayscale images
    - kernel: numpy.ndarray, shape (kh, kw), with the kernel for the conv.
    """
    step_h = kernel.shape[0]
    step_w = kernel.shape[1]
    window_h = images.shape[1] - kernel.shape[0] + 1
    window_w = images.shape[2] - kernel.shape[1] + 1
    m = images.shape[0]
    img = np.arange(m)
    convolution = np.zeros((m, window_h, window_w))
    for i in range(window_h):
        for j in range(window_w):
            convole = images[img, i:i + step_h, j:j + step_w] * kernel
            pos = np.sum(convole, axis=(1, 2))
            convolution[img, i, j] = pos

    return convolution

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
    m, h, w = images.shape
    kh, kw = kernel.shape
    out_w = w - kw + 1
    out_h = h - kh + 1
    convolve_out = np.ndarray((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolve_out[:, i, j] = np.sum(
                images[:, i:kh+i, j:kw+j] * kernel,
                axis=(1, 2)
                )

    return convolve_out

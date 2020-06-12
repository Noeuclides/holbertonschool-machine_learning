#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a valid convolution on grayscale images:
    - images: numpy.ndarray, shape (m, h, w), with multiple grayscale images
    - kernel: numpy.ndarray, shape (kh, kw), with the kernel for the conv.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = int((kh - 1) / 2)
    pw = int((kw - 1) / 2)

    if kh % 2 != 0:
        out_w = w + 2*pw - kw + 1
        out_h = h + 2*ph - kh + 1
        images_pad = np.pad(
            images, ((0, 0), (ph, ph), (pw, pw)), 'constant'
            )
    else:
        out_w = w + 2*pw - kw + 2
        out_h = h + 2*ph - kh + 2
        images_pad = np.pad(
            images, ((0, 0), (0, ph + 1), (0, pw + 1)), 'constant'
            )

    convolve_out = np.ndarray((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolve_out[:, i, j] = np.sum(
                images_pad[:, i:kh+i, j:kw+j] * kernel,
                axis=(1, 2))

    return convolve_out

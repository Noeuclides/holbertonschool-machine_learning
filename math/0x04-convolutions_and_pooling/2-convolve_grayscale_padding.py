#!/usr/bin/env python3
"""Convolution module
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:

    - images: numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
        - m: number of images
        - h: height in pixels of the images
        - w: width in pixels of the images
    - kernel: numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution
        - kh: height of the kernel
        - kw: width of the kernel
    - padding: tuple of (ph, pw)
        - ph: padding for the height of the image
        - pw: padding for the width of the image
        - the image should be padded with 0’s
    Returns: a numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    images_pad = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    out_w = w + 2*pw - kw + 1
    out_h = h + 2*ph - kh + 1

    convolve_out = np.ndarray((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            convolve_out[:, i, j] = np.sum(
                images_pad[:, i:kh+i, j:kw+j] * kernel,
                axis=(1, 2))

    return convolve_out

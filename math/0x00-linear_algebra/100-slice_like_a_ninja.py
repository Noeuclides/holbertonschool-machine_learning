#!/usr/bin/env python3
"""
Module to slice matrix on an axis
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along a specific axes
    """
    slices = []
    for axis in range(len(matrix.shape)):
        slices.append(slice(*axes.get(axis, (None, None, None))))

    return matrix[tuple(slices)]

#!/usr/bin/env python3
"""module with a method that calculates shape
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix
    """
    m = matrix[:]
    shape = []
    while isinstance(m, list):
        shape.append(len(m))
        m = m[0]

    return(shape)

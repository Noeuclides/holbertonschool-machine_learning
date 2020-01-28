#! /usr/bin/env python3
"""module to transpose
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    """
    inner = []
    transpose = []
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            inner.append(matrix[i][j])
        transpose.append(inner)
        inner = []

    return transpose

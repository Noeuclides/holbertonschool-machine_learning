#!/usr/bin/env python3
"""
Module to add matrices
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices
    """
    if len(mat1) != len(mat2):
        return None

    addMatrix = []
    for i in range(len(mat1)):
        if isinstance(mat1[i], list):
            resultAxis = add_matrices(mat1[i], mat2[i])
            if resultAxis is None:
                return None
            addMatrix.append(resultAxis)
        else:
            addMatrix.append(mat1[i] + mat2[i])

    return addMatrix

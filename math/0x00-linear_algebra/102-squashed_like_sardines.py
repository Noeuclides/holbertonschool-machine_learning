#!/usr/bin/env python3
"""
Module to concatenate matrices
"""


def cat(mat1, mat2, axis, a=1, new=[]):
    for index, elem in enumerate(mat1):
        if axis == a:
            m = elem + mat2[index]
            new.append(m)
        else:
            a += 1
            m = cat(elem, mat2[index], axis, a, new)
            new.append(m)

    return new


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


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two matrices
    """

    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    if axis == 0:
        return mat1 + mat2

    new_m = cat(mat1, mat2, axis, a=1, new=[])

    return new_m

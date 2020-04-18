#!/usr/bin/env python3
"""
Module to concatenate matrices
"""


def concat_recursive(mat1, mat2, axis, inner):
    """
    concatenates arrays recursevely
    """
    new = []
    for index, elem in enumerate(mat1):
        if axis == inner:
            m = elem + mat2[index]
            new.append(m)
        else:
            inner += 1
            new.append(concat_recursive(elem, mat2[index], axis, inner))
            inner -= 1

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

    shape1.pop(axis)
    shape2.pop(axis)

    if shape1 != shape2:
        return None

    if len(shape1) < axis:
        return None

    if axis == 0:
        return mat1 + mat2

    concat_mat = concat_recursive(mat1, mat2, axis, inner=1)
    return concat_mat

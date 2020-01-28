#!/usr/bin/env python3
"""
Module to concatenate matrices on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    cat = []
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for item in mat1:
            inner = item[:]
            cat.append(inner)

        for elem in mat2:
            cat.append(elem)

        return cat

    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            nsum = mat1[i] + mat2[i]
            cat.append(nsum)

        return cat

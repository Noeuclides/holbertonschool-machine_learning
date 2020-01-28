#!/usr/bin/env python3
"""
Module to concatenate matrices on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    cat = []
    for item in mat1:
        inner = item[:]
        cat.append(inner)

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for elem in mat2:
            cat.append(elem)
    elif axis == 1 and len(mat1) == len(mat2):
        for i in range(len(cat)):
            for j in range(1):
                cat[i].append(mat2[i][j])

    return cat

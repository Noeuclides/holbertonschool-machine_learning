#!/usr/bin/env python3
"""
Module to concatenate matrices on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    cat = mat1[:]
    for i in range(len(cat)):
        inner = cat[i][:]
        cat[i] = inner

    if axis == 0:
        for elem in mat2:
            cat.append(elem)
    else:
        for i in range(len(cat)):
            for j in range(1):
                cat[i].append(mat2[i][j])

    return cat

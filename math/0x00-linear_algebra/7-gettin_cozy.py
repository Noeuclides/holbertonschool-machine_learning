#!/usr/bin/env python3
"""
Module to concatenate matrices on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    cat = []
    cat2 = []
    for item in mat1:
        inner = item[:]
        cat.append(inner)

    if axis == 0 and len(cat[0]) == len(mat2[0]):
        for elem in mat2:
            cat.append(elem)
        return cat
    if axis == 1 and len(cat) == len(mat2):
        new = []
        for i in range(len(cat)):
            nsum = cat[i] + mat2[i]
            new.append(nsum)
        return new

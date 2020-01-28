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
    for item in mat2:
        inner2 = item[:]
        cat2.append(inner2)

    if axis == 0 and len(cat[0]) == len(cat2[0]):
        for elem in cat2:
            cat.append(elem)
        return cat
    if axis == 1 and len(cat) == len(cat2):
        new = []
        for i in range(len(cat)):
            nsum = cat[i] + cat2[i]
            new.append(nsum)
        return new

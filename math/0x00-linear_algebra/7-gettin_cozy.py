#!/usr/bin/env python3
"""
Module to concatenate matrices on axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    cat = mat1.copy()
    print("CAT: ", cat)
    if axis == 0:
        print("something")
        for elem in mat2:
            cat.append(elem)
    else:
        print("another")
        for i in range(len(cat)):
            for j in range(1):
                cat[i].append(mat2[i][j])

    print("mat1: ", mat1)

    return cat

#! /usr/bin/env python3


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise
    """
    if len(mat1) != len(mat2):
        return None
    elif len(mat1[0]) != len(mat2[0]):
        return None

    add = []
    total = []
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            result = mat1[i][j] + mat2[i][j]
            add.append(result)
        total.append(add)
        add = []

    return(total)

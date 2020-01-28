#!/usr/bin/env python3
"""
module to multiplicate matrices
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None

    item = 0
    inArray = []
    mulArr = []
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                item += mat1[i][k] * mat2[k][j]
            inArray.append(item)
            item = 0
        mulArr.append(inArray)
        inArray = []

    return mulArr

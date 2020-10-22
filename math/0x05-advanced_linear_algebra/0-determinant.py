#!/usr/bin/env python3
"""
module to get the determinant of a matrix
"""
import copy


def determinant(matrix: list) -> int:
    """
    calculates the determinant of a matrix:
    - matrix: list of lists whose determinant should be calculated
    Returns: the determinant of the matrix
    """
    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        return 1 if len(matrix[0]) == 0 else matrix[0][0]

    if not check_square(matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 2:
        return two_by_two_det(matrix)

    det = 0
    for j in range(len(matrix)):
        minor = minor_matrix(matrix, j)
        cofactor = (-1)**(j)
        det += matrix[0][j] * cofactor * determinant(minor)

    return det


def two_by_two_det(matrix: list) -> int:
    """
    gets the determinant of a 2x2 matrix:
    - matrix: 2x2 matrix
    return determinant
    """
    ad = matrix[0][0] * matrix[1][1]
    bc = matrix[1][0] * matrix[0][1]
    return ad - bc


def minor_matrix(matrix: list, col_to_del: int) -> list:
    """
    Gets the minor of a matrix
    - matrix: matrix to get the minor
    - col_to_del: column that has to be remove.
    return matrix minor
    """
    mat = copy.deepcopy(matrix)
    # deletes the first row of the matrix
    del mat[0]
    # loop for delete the column in the index col_to_del
    for i in range(len(mat)):
        del mat[i][col_to_del]

    return mat


def check_square(matrix: list) -> bool:
    """
    check if it's a square matrix
    - matrix: matrix to be checked
    return True if it's a square matrix, False if not
    """
    for elem in matrix:
        if len(elem) != len(matrix):
            return False

    return True

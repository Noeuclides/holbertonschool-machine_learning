#!/usr/bin/env python3
"""
module to get the inverse matrix of a matrix
"""


def determinant(matrix: list) -> int:
    """
    calculates the determinant of a matrix:
    - matrix: list of lists whose determinant should be calculated
    Returns: the determinant of the matrix
    """
    if len(matrix) == 1:
        if len(matrix[0]) == 0:
            return 1
        elif len(matrix[0]) == 1:
            return matrix[0][0]
    for elem in matrix:
        # check if it's a square matrix
        if len(elem) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 2:
        return two_by_two_det(matrix)

    det = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            minor = minor_matrix(matrix, i, j)
            cofactor = (-1)**(i + j)
            det += matrix[0][j] * cofactor * determinant(minor)

    return det


def recursive_minor(matrix: list) -> list:
    """
    calculates the minor matrix of a square matrix with side bigger than 2
    recursevely:
    - matrix: list of lists whose minor matrix should be calculated
    Returns: the minor matrix of matrix
    """
    if len(matrix) == 2:
        return two_by_two_det(matrix)

    minor_det = []
    for i in range(len(matrix)):
        inner = []
        det = 0
        for j in range(len(matrix[i])):
            sub_minor = minor_matrix(matrix, i, j)
            cofactor = (-1)**(i + j)
            inner.append(determinant(sub_minor) * cofactor)
        minor_det.append(inner)
    return minor_det


def inverse(matrix: list) -> list:
    """
    calculates the inverse of a matrix:
    - matrix: list of lists whose cofactor matrix should be calculated
    Returns: the inverse of matrix, or None if matrix is singular
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for elem in matrix:
        if not isinstance(elem, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1:
        if len(matrix[0]) == 1:
            return [[1 / matrix[0][0]]]
        elif len(matrix[0]) == 0:
            raise ValueError("matrix must be a non-empty square matrix")
    for elem in matrix:
        # check if it's a square matrix
        if len(elem) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    # check if matrix is singular
    if det == 0:
        return None

    if len(matrix) == 2:
        return [
            [matrix[1][1] / det, -1 * matrix[0][1] / det],
            [-1 * matrix[1][0] / det, matrix[0][0] / det]
            ]

    inverse = []
    cofactor = recursive_minor(matrix)
    for i in range(len(cofactor)):
        inner = []
        for j in range(len(cofactor[i])):
            inner.append(cofactor[j][i] * (1 / det))
        inverse.append(inner)

    return inverse


def two_by_two_det(matrix: list) -> int:
    """
    gets the determinant of a 2x2 matrix:
    - matrix: 2x2 matrix
    return determinant
    """
    ad = matrix[0][0] * matrix[1][1]
    bc = matrix[1][0] * matrix[0][1]
    return ad - bc


def minor_matrix(matrix: list, row: int, col: int) -> list:
    """
    Gets the minor of a matrix
    - matrix: matrix to get the minor
    - row: row matrix that has to be removed
    - col: column that has to be remove.
    return matrix minor
    """
    mat = deep_copy(matrix)
    # deletes the first row of the matrix
    del mat[row]
    # loop for delete the column in the index col_to_del
    for i in range(len(mat)):
        del mat[i][col]

    return mat


def deep_copy(matrix: list) -> list:
    """
    makes a deep copy of a matrix
    - matrix: matrix to make a deep copy
    return deep copy
    """
    new_mat = []
    for i in range(len(matrix)):
        inner = []
        for elem in matrix[i]:
            inner.append(elem)
        new_mat.append(inner)

    return new_mat

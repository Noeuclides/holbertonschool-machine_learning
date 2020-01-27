#! /usr/bin/env python3

def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix
    """
    m = matrix[:]
    shape = []
    while type(m) is list:
        shape.append(len(m))
        m = m[0]
    
    return(shape)


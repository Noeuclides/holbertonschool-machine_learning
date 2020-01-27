#! /usr/bin/env python3

def matrix_shape(matrix):
    """
    Calculates the shape of a matrix
    """
    m = matrix[:]
    shape = []
    while type(m) is list:
        shape.append(len(m))
        m = m[0]
    
    return(shape)


#!/usr/bin/env python3
"""
module to derivate
"""

def poly_derivative(poly):
    """
    calculate the derivative of a polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivate = [coef[0] * coef[1] for coef in enumerate(poly)]
    if derivate[0] == 0:
        derivate.pop(0)
    
    return derivate

#!/usr/bin/env python3
"""
module to integrate
"""

def poly_integral(poly, C=0):
    """
    calculate the integral of a polynomial
    """
    if len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    integral = [coef[1] / (coef[0]+1) for coef in enumerate(poly)]
    integral.insert(0, C)
    
    return integral

#!/usr/bin/env python3
"""
module to integrate
"""

def poly_integral(poly, C=0):
    """
    calculate the integral of a polynomial
    """
    if not poly:
        return None

    if len(poly) == 1:
        return [0]

    integral = [C]
    for coef in enumerate(poly):
        c = coef[1] / (coef[0] + 1)
        if c % 1 == 0:
            integral.append(int(c))
        else:
            integral.append(c)
        
    
    return integral

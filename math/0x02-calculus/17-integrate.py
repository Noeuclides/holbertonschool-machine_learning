#!/usr/bin/env python3
"""
module to integrate
"""

def poly_integral(poly, C=0):
    """
    calculate the integral of a polynomial
    """
    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None
   
    if C % 1 == 0:
        integral = [int(C)]
    else:
        integral = [C]
    for coef in enumerate(poly):
        c = coef[1] / (coef[0] + 1)
        if c % 1 == 0:
            integral.append(int(c))
        else:
            integral.append(c)
        
    
    return integral

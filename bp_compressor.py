#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:38:16 2018

@author: diego
"""

from bp_reader import bp_reader
from SHOPOD import SHOPOD
import high_order_decomposition_method_functions as hf
import numpy as np

def bp_compressor(Variable, Principal_Variable_number=0, tol=1e-7):
    """
    In this code we are going to compress the variable data 
    contained in a bp folder using the SHOPOD as a reduction method
    with a unitary matrix as mass matrices to simulate the result of 
    high order SVD method to avoid the use of the grid.\n
    Parameter: \n
    Variable: string type, the name of the variable to extract is spected
    such as "pressure", "density", "vitesse" etc. \n
    Principal_Variable_number: If there are several simulations in the 
    same file (simulations for differents cases) adios will create 
    differents files, this code in order to first example will take only
    one of this simulations, default value will be 0 as there is only
    going to be at least one simulation. \n
    tol= This variable represents an error estimation that is taken from
    eigen values of the correlation Matrix, this is not the final error.
    A tolerance in the order of 1e-2 will generate results with much
    lower errors errors values.
    """
    
    file=bp_reader(Variable)
    Mass=hf.unit_mass_matrix_creator(file[Principal_Variable_number])
    Reduced=SHOPOD(file[Principal_Variable_number],Mass,tol=tol)
    
    return file[Principal_Variable_number], Reduced
    
    
    
    
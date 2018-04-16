# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:17:19 2018

@author: Diego Britez
"""

import numpy as np
from scipy.sparse import diags
import integrationgrid

def tosparse(X):
    """
    This function serve to transform a list of two arrays with the grid points
    information in two sparse diagonals matrices with the integration points 
    for the trapezoidal integration method
    """
    dim=2
    tshape=np.array([X[0].size,X[1].size])
    M=integrationgrid.IntegrationGrid(X,dim,tshape)
    M=M.IntegrationGridCreation()
    Mx=diags(M[0])
    Mt=diags(M[1])
    
    return Mx,Mt
    
    
    
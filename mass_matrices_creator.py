# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:17:19 2018

@author: Diego Britez
"""

import numpy as np
from scipy.sparse import diags
import integrationgrid

def mass_matrices_creator(X):
    """
    This function serve to transform a list of two arrays with the grid points
    information in  sparse diagonals matrices with the integration points 
    for the trapezoidal integration method
    """
    
    dim=len(X)
    tshape=[]
    for i in range(dim):
        aux=X[i].size
        tshape.append(aux)
    #tshape=np.array([X[0].size,X[1].size])
    M=integrationgrid.IntegrationGrid(X,dim,tshape)
    M=M.IntegrationGridCreation()
    for i in range (dim):
        M[i]=diags(M[i])    
    #Mx=diags(M[0])
    #Mt=diags(M[1])
    
    return M
    
    
    
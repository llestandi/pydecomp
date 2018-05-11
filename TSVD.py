# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:19:09 2018

@author: Diego Britez
"""
import numpy as np
from scipy.sparse import diags
def TSVD(F, epsilon = 1e-10):
    """
    This function calculates a matrix decomposition by using the truncated SVD 
    method.\n
    The decomposition of the F matrix has following forme:\n
    :math:`F=U.\sigma.A^{t}` \n
    where U and A are the first and second subspaces projection of the matrix.\n
    :math:`\sigma` a square matrix with the singular values as the diagonal 
    elements.\n
    **Parameters**\n
    F= 2 dimention ndarray type of data (Matrix).\n
    epsilon= maximal value for the sigular value. Default value= 1e-10.\n
    **Returns**\n
    U: 2d array type \n
    :math:`\sigma` : sparse diagonal matrix type.\n
    A: 2d array type.\n
    
    
    """
    U, S, V = np.linalg.svd(F, full_matrices=True)
    aux=S[0]
    i=0
    imax=len(S)
    while (aux>epsilon) & (i<imax):
        aux=S[i]
        i+=1
        srank=i
        
    s=S[:srank]
    s=diags(s)
    u=U[::,:srank]
    V=V.T
    v=V[::,:srank]
    return u,s,v
    
 
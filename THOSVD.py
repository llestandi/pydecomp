# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:52:54 2018

@author: Diego Britez
"""
from TSVD import TSVD
import high_order_decomposition_method_functions as hf
from Tucker import Tucker

def THOSVD(F):
    """
    This method decomposes a ndarray type data (multivariable) in a Tucker 
    class element by using the Tuncated High Order Singular Value Decomposition
    method.\n
    **Paramameter**\n
    F: ndarray type element.\n
    **Returns**
    Decomposed_Tensor: Tucker class type object. To read more information about
    this object type, more information could be found in Tucker class
    documentation.
    
    
    """
    
    tshape=F.shape
    dim=len(tshape)
    PHI=[]
    for i in range(dim):
        Fmat=hf.matricize(F,dim,i)
        phi,sigma,A=TSVD(Fmat)
        PHI.append(phi)
    PHIT=hf.list_transpose(PHI)
    W=hf.multilinear_multiplication(PHIT,F,dim)
    Decomposed_Tensor=Tucker(W,PHI)
    
    return Decomposed_Tensor
        
 #-----------------------------------------------------------------------------       

    

# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:00:48 2018

@author: Diego Britez
"""
from TSVD import TSVD
import high_order_decomposition_method_functions as hf
from Tucker import Tucker
import numpy as np


def STHOSVD(F):
    """
    This method decomposes a ndarray type data (multivariable) in a Tucker 
    class element by using the Secuentialy Tuncated High Order Singular Value
    Decomposition method.\n
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
    W=F
    for i in range(dim):
        Wshape=[x for x in W.shape]
        Wmat=hf.matricize(W,dim,0)
        phi,sigma,A=TSVD(Wmat)
        W=sigma@A.T
               
        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        W=np.moveaxis(W,0,-1)
        
        PHI.append(phi)
        
    Decomposed_Tensor=Tucker(W,PHI)
    
    return Decomposed_Tensor
#------------------------------------------------------------------------------
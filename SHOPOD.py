# -*- coding: utf-8 -*-
"""
Created on Mon May  7 11:43:45 2018

@author: Diego Britez
"""
from POD import POD2 as POD
import high_order_decomposition_method_functions as hf
from Tucker import Tucker
from scipy.sparse import diags
import numpy as np

def SHOPOD(F,MM, tol=1e-5):
    """
    This method decomposes a ndarray type data (multivariable) in a Tucker 
    class element by using the Secuentialy  High Order Proper Orthogonal 
    Decomposition method.\n
    **Paramameter**\n
    F: ndarray type element.\n
    **MM**:list of mass matrices (integration points for trapeze integration 
    method) as sparse.diag type elements\n
    **Returns**
    Decomposed_Tensor: Tucker class type object. To read more information about
    this object type, more information could be found in Tucker class
    documentation.
    """
    M=MM[:]
    tshape=F.shape
    dim=len(tshape)
    PHI=[]
    W=F
    for i in range(dim):
        Wshape=[x for x in W.shape]
        Wmat=hf.matricize(W,dim,0)
        Mx,Mt = hf.matricize_mass_matrix(dim,0,M)
        phi,sigma,A=POD(Wmat.T,Mt,Mx,tol=tol)
        W=sigma@phi.T
        
               
        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        #cambiar matriz de masa y enviarlo al final
        M[0]=diags(np.ones(Wshape[0]))
        M.insert((dim-1),M.pop(0))
        W=np.moveaxis(W,0,-1)
        
        PHI.append(A)
        
    Decomposed_Tensor=Tucker(W,PHI)
    
    return Decomposed_Tensor
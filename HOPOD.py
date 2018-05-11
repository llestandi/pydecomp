# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:46:04 2018

@author: Diego Britez
"""


from POD import POD2 as POD
import high_order_decomposition_method_functions as hf
from Tucker import Tucker


def HOPOD(F,X):
    """
    
    Returns a decomposed tensor in the tucker format class.\n
    
    The projection of a tensor in each dimention is calculated with the 
    POD method, to achieve this operation a matricitization of the tensor is 
    carried out for each dimention, so the orthogonal projection is found
    thanks to this apparent 2D problem in each step. \n
    
    **Parameters:**\n
        **F:** Tensor  of n dimentions. Array type.\n
        **X:** Cartesian grid which describes data distribution of F in space.
        List of array type elements.\n
        
    **Returns:** \n
        Tucker class element\n
        
        
    """
    
    
    tshape=F.shape
    dim=len(tshape)
    M=hf.mass_matrices_creator(X)
    PHI=[]
    
    for i in range(dim):
        Fmat=hf.matricize(F,dim,i)   
        Mx,Mt = hf.matricize_mass_matrix(dim,i,M)
        phi,sigma,A= POD(Fmat.T,Mt,Mx)
        PHI.append(A) 
    PHIT=hf.list_transpose(PHI)
    PHIT=hf.integrationphi(PHIT,M) 
    W =hf.multilinear_multiplication(PHIT, F, dim)
    Decomposed_Tensor=Tucker(W,PHI)
    
    
    return Decomposed_Tensor
    

   








    
   
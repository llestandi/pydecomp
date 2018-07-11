# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:46:04 2018
@author: Diego Britez

Fusion of all tucker related decomposition methods performed by Lucas on 28/06/18
"""
from scipy.sparse import diags
import scipy.sparse
import numpy as np

from Tucker import TuckerTensor
from POD import POD
from TSVD import TSVD

import tensor_algebra as ta
import MassMatrices as mm
import sys
sys.path.append('/media/diego/OS/Users/Diego Britez/Documents/stage/pydecomp_V2/pydecomp/utils')
import misc as misc

def HOPOD(F,M, tol=1e-10, sparse=False):
    """
    Returns a decomposed tensor in the tucker format class.\n

    The projection of a tensor in each dimention is calculated with the
    POD method, to achieve this operation a matricitization of the tensor is
    carried out for each dimention, so the orthogonal projection is found
    thanks to this apparent 2D problem in each step. \n

    **Parameters:**\n
    **F**: Tensor  of n dimentions. Array type.\n
    **M**:list of mass matrices (integration points for trapeze integration
    method) as sparse.diag type elements\n

    **Returns:** \n
    Tucker class element\n
    """
    tshape=F.shape
    dim=len(tshape)
    PHI=[]
    if type(M[0])==scipy.sparse.dia.dia_matrix:
        SPARSE=True
    else:
        SPARSE=False

    for i in range(dim):
        Fmat=ta.matricize(F,dim,i)
        Mx,Mt = mm.matricize_mass_matrix(dim,i,M)
        phi,sigma,A= POD(Fmat.T,Mt,Mx, tol=tol)
        PHI.append(A)
    PHIT=misc.list_transpose(PHI)
    PHIT=integrationphi(PHIT,M,sparse=SPARSE)
    W =ta.multilinear_multiplication(PHIT, F, dim)
    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor

def integrationphi(PHIT,M,sparse=False):
   """
   This function integrates each projection (phi) by using the correspondent
   mass matrix.
   """
   
   a="""
   Dimentions of decomposition values list and Mass matrices list are not
   coherents
   """
   if len(PHIT)!= len(M):
       raise ValueError(print(a))
   if not sparse:
       integrated_phi=[phit*m for (phit,m) in zip(PHIT,M)]
   elif sparse:
       integrated_phi=[phit@m for (phit,m) in zip(PHIT,M)]
   return integrated_phi


def SHOPOD(F,MM, tol=1e-10,rank=-1):
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
    if type(M[0])==scipy.sparse.dia.dia_matrix:
        SPARSE=True
    else:
        SPARSE=False
    PHI=[]
    W=F
    for i in range(dim):
        Wshape=[x for x in W.shape]
        Wmat=ta.matricize(W,dim,0)
        Mx,Mt = mm.matricize_mass_matrix(dim,0,M)
        phi,sigma,A=POD(Wmat.T,Mt,Mx,tol=tol,rank=rank)
        
        W=(sigma*phi).T


        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        #changing the first mass matrix for a unit vector and send it to the 
        #back
        
        M[0]=np.ones(Wshape[0])
        if SPARSE:
            M[0]=diags(M[0])
        M.insert((dim-1),M.pop(0))
        W=np.moveaxis(W,0,-1)

        PHI.append(A)
    
    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor


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
        Wmat=ta.matricize(W,dim,0)
        phi,sigma,A=TSVD(Wmat)
        
        W=sigma@A.T

        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        W=np.moveaxis(W,0,-1)

        PHI.append(phi)

    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor

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
        Fmat=ta.matricize(F,dim,i)
        phi,sigma,A=TSVD(Fmat)
        PHI.append(phi)
    PHIT=misc.list_transpose(PHI)
    W=ta.multilinear_multiplication(PHIT,F,dim)
    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor

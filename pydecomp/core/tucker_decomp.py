# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:46:04 2018
@author: Diego Britez

Fusion of all tucker related decomposition methods performed by Lucas on 28/06/18
"""
from scipy.sparse import diags
import scipy.sparse
import numpy as np
import timeit
from core.Tucker import TuckerTensor
from core.POD import POD
from core.TSVD import TSVD
import core.tensor_algebra as ta
import core.MassMatrices as mm
from core.MassMatrices import identity_mass_matrix
import utils.misc as misc

from copy import deepcopy

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
    if M.is_sparse:
        SPARSE=True
    else:
        SPARSE=False

    for i in range(dim):
        Fmat=ta.matricize(F,dim,i)
        Mx,Mt = mm.matricize_mass_matrix(dim,i,M)
        phi,sigma,A= POD(Fmat.T,Mt,Mx, tol=tol)
        PHI.append(A)
    PHIT=misc.list_transpose(PHI)
    W=tucker_weight_eval(PHIT,M,F,dim,sparse=SPARSE)
    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor

def tucker_weight_eval(PHIT,MM,F,dim,sparse=False):
    """
    This function reproduces the operation:\n
    :math:`	(\phi_{1},\phi_{2},...,\phi{d})[(M_{1},M_{2},...,M_{d}).
    \mathcal{F}]=(\phi_{1}M_{1},\phi_{2},M_{2},...,\phi{d}M_{d}).\mathcal{F}`
    """
    a="""
    Dimentions of decomposition values list and Mass matrices list are not
    coherents
    """
    if len(PHIT)!= len(MM.Mat_list):
        raise ValueError(a)
    if not sparse:
        integrated_phi=[phit*m.M for (phit,m) in zip(PHIT,MM.Mat_list)]
    elif sparse:
        integrated_phi=[phit@m.M for (phit,m) in zip(PHIT,MM.Mat_list)]
    W =ta.multilinear_multiplication(integrated_phi, F, dim)
    return W


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
    from copy import deepcopy,copy
    M=deepcopy(MM)
    tshape=F.shape
    dim=len(tshape)
    PHI=[]
    W=F
    for i in range(dim):
        Wshape=[x for x in W.shape]
        Wmat=ta.matricize(W,dim,0)
        Mx,Mt = mm.matricize_mass_matrix(dim,i,M)
        phi,sigma,A=POD(Wmat.T,Mt,Mx,tol=tol,rank=rank)
        W=(sigma*phi).T
        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        #changing the first mass matrix for a unit vector and send it to the
        #back
        M.update_mass_matrix(i, identity_mass_matrix(Wshape[0],M.is_sparse))
        # M.insert((dim-1),M.pop(0))
        W=np.moveaxis(W,0,-1)
        PHI.append(A)

    Decomposed_Tensor=TuckerTensor(W,PHI)
    return Decomposed_Tensor


def STHOSVD(F,epsilon = 1e-13, rank=100, solver='EVD'):
    """
    This method decomposes a ndarray type data (multivariable) in a Tucker
    class element by using the Secuentialy Tuncated High Order Singular Value
    Decomposition method.\n
    **Paramameter**\n
    F: ndarray type element.\n
    epsilon: stoping criteria of the algorithm.\n
    rank: maximal number of ranks if the stoping criteria was not reached.\n
    solver:
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
        phi,sigma,A=TSVD(Wmat, epsilon=epsilon, rank=rank, solver=solver)
        W=(sigma*A).T
        Wshape[0]=W.shape[0]
        W=W.reshape(Wshape)
        W=np.moveaxis(W,0,-1)
        PHI.append(phi)
    Decomposed_Tensor=TuckerTensor(W,PHI)
    return Decomposed_Tensor


def THOSVD(F,epsilon = 1e-13, rank=100, solver='EVD'):
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
        phi,sigma,A=TSVD(Fmat, epsilon=epsilon, rank=rank, solver=solver)
        PHI.append(phi)
    PHIT=misc.list_transpose(PHI)
    W=ta.multilinear_multiplication(PHIT,F,dim)
    Decomposed_Tensor=TuckerTensor(W,PHI)

    return Decomposed_Tensor

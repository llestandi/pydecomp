# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:30:18 2018

@author: Diego Britez

"""
# @TODO Likely to endup in full tensor file
# @Diego TODO Incorporate wrapper for norm and scalar product easy computing
# (from multilinear multiplications, especially on ndarrays)

import numpy as np
import scipy.sparse
from scipy.sparse import diags

def multilinear_multiplication(PHI,F,dim):
    """
    **Parameters**:_ \n

    phi= Is a list with matrices type elements as the elements.\n 
    F  = nd Array type . \n
    dim = tuple, list or array type that its value must be coherent with
    the number of elements of phi. \n \n
    **Return**:_ \n

    W= Array type element, result of an unfolded multilinear multiplication. \n

    **Definition** \n
    In general the *unfolding multilinear multiplication* is given by: \n
    :math:`[(\phi_{1},\phi_{2},...,\phi_{n}).F]^{n}=\phi_{n}F^{n}
    (\phi_{1} \otimes ... \otimes \phi_{m-1} \otimes \phi_{m+1} \otimes ...
    \otimes \phi_{d})^{T}` \n
    where
    :math:`\otimes` represents the Kronecker product.
    """
    number_modes=np.array([x.shape[0] for x in PHI])
    maximal_index=np.argmax(number_modes)
    PHI2=PHI[:]
    PHI2.insert(0,PHI2.pop(maximal_index))
    Fmat=matricize(F,dim,maximal_index)     
    aux=PHI2[1]
    for i in range(1,dim-1):
        aux=np.kron(aux,PHI2[i+1])
    aux=aux.T
    MnX=PHI2[0]@Fmat
    W=MnX@aux
    forme_W=[i.shape[0] for i in PHI2]
    W = W.reshape(forme_W)
    W  =np.moveaxis(W,0,maximal_index)
    return W


def kathri_rao(A,B):
    """
    kathri_rao product of matrices A (IxK) and B (JxK) which result in a matrix
    of size (IJxK). See Kolda and Balder definition.
    """
    Ka=A.shape[1]
    Kb=B.shape[1]
    if Ka==Kb:
        K=Ka
    else:
        raise AttributeError("A.shape[1]= {0} != {1} B.shape[1], Kathri-rao\
                             product is not allowed".format(Ka,Kb))
    return np.transpose(np.stack([np.kron(A[:,i],B[:,i]) for i in range(K) ]))

def multi_kathri_rao(matrices):
    """Computes the kr product of all matrices in the list"""

    if type(matrices)==np.ndarray:
        return matrices
    if len(matrices)==1:
        return matrices[0]

    prod=np.ones((1,matrices[0].shape[1]))
    for M in matrices:
        prod=kathri_rao(prod,M)

    return prod

#------------------------------------------------------------------------------
def matricize(F,dim,i):
    """
    Returns the tensor F rearanged as a matrix with the *"i"* dimention as
    the new principal (order 0) order. In other words it returns the unfolded
    tensor as a matrix with the "i" dimention as the new "0" dimention.  \n
    **Parameters**\n
    F= array type of n dimentions.\n
    dim= number of dimentions of the tensor F.\n
    i= dimention that is going to be taken as the principal to matricize.
    """
    F1=np.moveaxis(F,i,0)
    F1shape=F1.shape
    F1=F1.reshape((F1shape[0],np.prod(F1shape[1:])))
    return F1
#------------------------------------------------------------------------------
def truncate_modes(Lambda,tol,rank,U):
    """
    This function evaluates the values of eingenvalues comparing to the maximal
    tolerance or the maximal number of rank(modes) in order to avoid nan values
    and unnecesary calcul, the final number of modes will be reduced.
    """
    imax=len(Lambda)
    if rank>=0:
        imax=min(len(Lambda),rank)

    Lambda1=Lambda[0]
    i=0
    stop_criteria=1
    while (stop_criteria>tol) & (i<imax) :
       stop_criteria=abs(Lambda[i]/Lambda1)
       i+=1
    Lambda=Lambda[:i]
    U=U[::,:i]
    return Lambda, U

def normeL2(T,M):
    """
    This function returns the norm defined in :math:`L^{2}`\n
    **Parameters**\n
    T= ndarray type. Tensor which its norme is going to be evaluated.
    M= list type with ndarrays elements. It's elements should be the integration
    points.     
    """
    dim=len(T.shape)
    dim2=len(M)
    if dim!=dim2:
        a="""
        Dimentions of the mass list is not coherent with the tensor dimension
        number
        """
        raise AttributeError(a)
    Rep=multilinear_multiplication(M,T,dim)
    Rep=T.T@Rep
    return Rep
    
#------------------------------------------------------------------------------
if __name__=="__main__":
    A=np.random.rand(3,4)
    B=np.random.rand(5,4)
    C=np.random.rand(2,4)
    kr=kathri_rao(A,B)
    print(kr.shape)
    print(kr)
    print("multiple kathri rao")
    print(multi_kathri_rao([A,B,C]))

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
from pydecomp.core.MassMatrices import MassMatrices, DiaMatrix, Kronecker
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
    if F.size > 1e6:
        print("Very large multilinear multiplication of size :")
        print(F.shape)
    if type(PHI)==MassMatrices:
        PHI2=[PHI.Mat_list[i] for i in range(len(PHI))]
    else :
        PHI2=PHI[:]
    W=F
    shape_W=list(W.shape)
    for i in range(dim):
        shape_W[0],shape_W[i]=shape_W[i],shape_W[0]
        W=np.swapaxes(W,0,i)
        F_mat=matricize(W, 0) 
        #this copy greatly improves the speed of the multilinear multiplication at very little cost
        phi=np.copy(PHI2[i])
        W=phi @ F_mat
        #W=PHI2[i]@matricize(W, 0)
        shape_W[0]=W.shape[0]
        W = W.reshape(shape_W)
        
    W  =np.moveaxis(W,0,-1)
    return W

def transpose(M):
    if type(M)==DiaMatrix :
        return M.transpose()
    else:
        return np.transpose(M)

def kron(A,B):
    if type(A)==DiaMatrix and type(A)==type(B):
        return Kronecker(MassMatrices([A,B]))
    else:
        return np.kron(A,B)

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
def matricize(F,i):
    """
    Returns the tensor F rearanged as a matrix with the *"i"* dimention as
    the new principal (order 0) order. In other words it returns the unfolded
    tensor as a matrix with the "i" dimention as the new "0" dimention.  \n
    **Parameters**\n
    F= array type of n dimentions.\n
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
        if Lambda[i]<0 :
            print("Warning, detected negative lambda, breaking")
            break
        i+=1
    Lambda=Lambda[:i]
    U=U[::,:i]
    return Lambda, U

def norm(T,MM=None,type="L2"):
    if type=="L2":
        if MM:
            return normL2(T,MM)
        else:
            return np.linalg.norm(T)
    elif type=="L1":
        if MM:
            return normL1(T,MM)
        else:
            return np.linalg.norm(np.ravel(T),ord=1)
    elif type=="Linf":
        return normLinf(T)
    else:
        raise TypeError("Wrong type {} give. Must be L1, L2 or Linf".format(type))

def scal_prod_1d(Phi,T,M,i):
    """1D scalar product between Phi (a matrix), T (a tensor) with integration
    rule M at index i"""
    if Phi.shape[0] != T.shape[i]:
        raise AttributeError("Phi ({}) and T ({}) dimension do not match\
                             ".format(Phi.shape[0],T.shape[i]))
    if Phi.shape[0] != M.dia_size:
        raise AttributeError("Phi ({}) and M ({}) dimension do not match\
                             ".format(Phi.shape[0],M.diag_size))
    r=Phi.shape[1]
    shape=list(T.shape)
    new_shape=shape[:i]+[r]+shape[i+1:]
    return np.reshape(np.transpose(M@Phi) @ matricize(T,i),new_shape)


def scal_prod_mem_save(Phi_list, T, MM):
    """ Matrix list against tensor product weighted by integration matrices MM """
    buff=T
    for i in range(T.ndim):
        buff=scal_prod_1d(Phi_list[i],buff,MM.Mat_list[i],i)
    return buff

def scal_prod_full_T_weighted(X,Y,MM):
    """ This is the scalar product between tensors X and Y weighted by  MM """
    d=X.ndim
    if np.any(X.shape != Y.shape):
        raise AttributeError( "X and Y shape don't align, {} {}".format(X.shape, Y.shape))

    return np.sum(multilinear_multiplication(MM,X*Y,d))


def normL2(T,M):
    """
    This function returns the norm defined in :math:`L^{2}`\n
    **Parameters**\n
    T= ndarray type. Tensor which its norme is going to be evaluated.
    M= MassMatrices , i.e. integration rules
    """
    dim=len(T.shape)
    dim2=len(M)
    if dim!=dim2:
        raise AttributeError("Dimentions of the mass list is not coherent with \
                             the tensor dimension number {} /= {}".format(dim, dim2))

    return np.sqrt(scal_prod_full_T_weighted(T,T,M))

def normL1(T,M):
    """
    This function returns the norm defined in :math:`L^{1}`\n
    **Parameters**\n
    T= ndarray type. Tensor which its norme is going to be evaluated.
    M= MassMatrices , i.e. integration rules
    """
    dim=len(T.shape)
    dim2=len(M)
    if dim!=dim2:
        raise AttributeError("Dimentions of the mass list is not coherent with \
                             the tensor dimension number {} /= {}".format(dim, dim2))
    return np.sqrt(scal_prod_full_T_weighted(T,np.ones(T.shape),M))

def normLinf(T):
    return np.max(np.abs(T))

def vector_outer_product(vectors):
    import string
    subscripts = string.ascii_letters[:len(vectors)]
    subscripts = ','.join(subscripts) + '->' + subscripts
    # expands to `numpy.einsum('a,b,c,d,e->abcde', v[0], v[1], v[2], v[3], v[4])`
    return np.einsum(subscripts, *vectors)

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

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018
@author: Diego Britez
Major # REVIEW:  on 28/06/18
@author: Lucas Lestandi
"""
import numpy as np
from scipy.linalg import norm
from core.Canonical import CanonicalTensor
import utils.CartesianGrid as cg
import core.tensor_algebra as ta
import timeit
import core.MassMatrices

def PGD(M,F, epenri=1e-10, maxfix=10):
    """
    This function use the PGD method for a multivariate problem decomposition,
    returning an Canonical class objet.
    \n
    **Parameters**:\n
    **M**  MassMatrices object.\n
    method) as sparse.diag type elements\n
    **F**: ndarray type, it's the tensor that its decomposition is wanted.\n
    **epenri**: Stop criteria value for the enrichment loop, default value=1e-10.
    This value is obtained by the divition  of the first mode with the
    mode obtained in the last fixed-point iteration of the last variable
    (wich carries all the information of the module of the function),  more
    detailed information can be find in  Lucas Lestandi's thesis.

    **maxfix**:is the maximal number of iteration made inside of the fix point
    function before declare that the method has no convergence.
    Default value=15 \n

    **Introduction to the PGD method**\n

    Be :math:`F` the tensor that is going to be decomposed by the PGD method.
    And be
    :math:`u^{p}`
    the solution, so we can express:\n
    :math:`u^{p}=\sum_{m=1}^{p}\prod_{i=1}^{D}X_{i}^{p}(x_{i})`.\n
    The weak  formulation of the problem can be expressed as:\n
    :math:`\int_{\Omega}u^{*}(u-F)=0`, is the test function, \n
    for all :math:`u^{*} \in H^{1}(\Omega)`. If we take;\n
    :math:`u^{*}=\prod_{i=1}^{s-1} X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})`,\n
    the weak formulation becomes:\n
    :math:`\int_{\Omega}[\prod_{i=1}^{s-1}X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})(u^{p-1}+\prod_{i=1}^{s}X_{i}^{k+1}
    (x_{i})\prod_{i=s+1}^{D}X_{i}^{k}(x_{i})-F)]=0` \n
    \n

    that for convenience it could be expressed like; \n
    \n
    :math:`\\alpha^{s}\int_{\Omega_{s}}X^{*}(x_{s})X_{s}^{k+1}(x_{s})dx_{s}=
    -\int_{\Omega_{s}}X^{*}\sum_{j=1}^{p-1}(\\beta^{s}(j)X_{s}^{j})
    dx_{s}+\int_{\Omega_{s}}X^{*}\gamma^{s}(x_{s})dx_{s}`\n
    where
    :math:`\\alpha , \\beta , \gamma` are variables deduced from the precedent
    equation and that are expoded in the fix point variables section.

    """
    start=timeit.default_timer()
    tshape=F.shape
    dim=len(tshape)
    #Start value of epn that allows get in to the enrichment loop for the first iteration
    eps=1
    n_iter=0    
    C=CanonicalTensor(tshape,dim)
    C.solution_initialization()

    while (eps>=epenri):
        R=fixed_point(M,C._tshape,C._U,F,n_iter ,C._rank,maxfix)
        C.add_enrich(R)
        if C.get_rank()==1:
            REF=norm(R[dim-1])

        R_norm=norm(R[dim-1])
        eps=R_norm/REF
   
    #Eliminating the first (zeros) row created to initiate the algorithm
    C._U=[x[1:,::] for x in C._U]
    stop=timeit.default_timer()
    print(stop-start)
    return C

def fixed_point(M,tshape,U,F,z,r,max_iter):
    
    """
    This function calculates the n mode for each iteration in the
    enrichment loop for the PGD method. The definition of each variable
    (such as alpha, betha, etc) that it's used here could be found in detail
    in the manuscript of  Lucas Lestandi doctoral thesis.\n
    **Parameters** \n
    X= Cartesian Grid.\n
    tshape= Tensor shape. \n
    U= (n-1) modes of the decomposition. \n
    F= Tensor to be decomposed. \n
    z= No convergence times counter variable. \n
    r= Actual rank in the decomposition. \n
    maxfix= maximun number of iterations in point fix algorithm.\n
    **Returns** \n
    R = n mode of the decomposition in the PGD decomposition method.\n
    z = actualized number of iteration that did not converge, so: \n
    :math:`z_{n}=z_{n-1}+1` if the actual mode does not converge or \n
    :math:`z_{n}=z_{n-1}`
    if it has converged.
    """
    dim=np.size(tshape)
    R=[np.array([np.ones(x)]) for x in tshape]
    New_Solution=R[:]
    
    Old_Solution=New_Solution
    k=0
    eppf=1e-8  # @TODO @Diego Variable naming should say what it does (without comment)
    epsilon=1

    while ((epsilon>=eppf) & (k<max_iter)):
        k=k+1
        Old_Solution=R[dim-1]

        for i in range(dim):
            Alpha=alpha(R,M,i,dim)
            Gamma=gamma(R,F,M,i,dim)       
            Betha=beta(M,R,U,i,dim)
            aux=np.dot(U[i].T,Betha)
            aux=np.transpose(aux)
            R[i]=(-aux+Gamma)/Alpha
            
            if (i<(dim-1)):
                R[i]=R[i]/(norm(R[i]))

        epsilon=norm(R[dim-1]-Old_Solution)/norm(Old_Solution)
   
        
    return  R


def alpha(R,M, i, dim):
    """
    Gamma is a variable used in the fixed point algorithm that solves the 
    PGD method.\n
    R: Is a list of numpy arrays as its elements. Chaque element represents
    the solution of the current mode treated in the PGD method, so the number
    of elements of each array must be coincident with the discretisation of 
    the respective subspace.\n
    M: MassMatrices object.\n
    i:the indice of the dimention that is treated in  this loop.\n
    dim: number of dimention of the problem.\n
    This function calculates the value of the
    :math:`\\alpha`
    variable in the fix point algorithm:  \n

    :math:`\\alpha^{s}=\int_{\Omega/\Omega_{s}}\prod_{i=1}^{s-1}(X_{i}^{k+1})^
    {2}\prod_{i=s+1}^{d}(X_{i}^{k})^{2}`, which could be expressed also as: \n
    :math:`\\alpha^{s}=\prod_{i=1}^{s-1}\int_{\Omega_{i}}(X_{i}^{k+1})^
    {2}dx_{i}\prod_{i=s+1}^{d}\int_{\Omega_{i}}(X_{i}^{k})^{2}	dx_{i}`
    """
    R1=R[:]
    alpha=1
    R1[0],R1[i]=R1[i],R1[0]
    #M1=M[:]
    M1=M.DiaMatrix_list[:]
    M1=[m.M for m in M1]
    M1[0],M1[i]=M1[i],M1[0]
    M1=[m[:,np.newaxis] for m in M1]

    for j in range(1,dim):
        R1[j]=np.multiply(R1[j],R1[j])
        aux=R1[j]@M1[j]    
        alpha=alpha*aux
    return alpha


def gamma(R,F,M,i,dim):
    """
    Gamma is a variable used in the fixed point algorithm that solves the 
    PGD method.\n 
    **Parameters** \n
    R: Is a list of numpy arrays as its elements. Chaque element represents
    the solution of the current mode treated in the PGD method, so the number
    of elements of each array must be coincident with the discretisation of 
    the respective subspace.\n
    F: numpy array element. Is the full tensor that its decomposition is 
    wanted. \n
    M: MassMatrices object.\n
    i:the indice of the dimention that is treated in  this loop.\n
    dim: number of dimention of the problem.\n
    F2: numpy array element, a vector that is function of the 'i' dimention, so
    the number of elemens of this array will be equal to the number of elements
    of any vector discretized in the 'i' dimention.
    This function will return the value of the
    :math:`gamma
    variable for each iteration in the fix poitn algorithm.
    :math:`\gamma^{s}(x_{s})= \int_{\Omega/\Omega_{s}}(\prod_{i=1}^{s-1}
    X_{i}^{k+1}\prod_{i=s+1}^{D}X_{i}^{k}).F`
    """
    F2=F
    F2=np.swapaxes(F2,0,i)
    F2_shape=F2.shape
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]
    #M1=M[:]
    M1=M.DiaMatrix_list[:]
    M1=[m.M for m in M1]
    M1[0],M1[i]=M1[i],M1[0]
     
    #@TODO improve this by removing loop (use Kronecker and matmul)
    for j in range(1,dim):

       F2=np.multiply(F2,R1[j])
       F2=F2@M1[j]
       
    #@Lucas The other version using Kronecker and matmul, is not more efficient,
    # its equivalent but I think its harder to understand how it works. 
    
    #integrated_r1=[r1*m1 for (r1,m1) in zip(R1[1:],M1[1:])]
    #serie_kron=integrated_r1[0]
    #for i in range(dim-2):
    #    serie_kron=np.kron(serie_kron,integrated_r1[i+1])

    #F2=F2.reshape([F2_shape[0],np.prod(F2_shape[1:])])
    #F2=serie_kron@F2.T
    return F2    
    
def beta(M,R,U,i,dim):
    """
    Gamma is a variable used in the fixed point algorithm that solves the 
    PGD method.\n 
    M: MassMatrices object.\n
    R: Is a list of numpy arrays as its elements. Chaque element represents
    the solution of the current mode treated in the PGD method, so the number
    of elements of each array must be coincident with the discretisation of 
    the respective subspace.\n
    U: Is a list of numpy arrays as its elements. Chalement contains all the
    modes that had been solved until the actual iteration.\n
    i:the indice of the dimention that is treated in  this loop.\n
    dim: number of dimention of the problem.\n
    :math:`\\beta`
    variable in the fix point algorithm:  \n

    :math:`\\beta^{s}(j)=\int_{\Omega/\Omega_{s}}\prod_{i=1}^{s-1}(X_{i}^{k+1}
    X_{i}^{j})\prod_{i=s+1}^{d}(X_{i}^{k}X_{i}^{j})`, which  could be expressed
    as, \n
    :math:`\\beta^{s}(j)=\prod_{i=1}^{s-1}\int_{\Omega}(X_{i}^{k+1}X_{i}^{j})
    \prod_{i=s+1}^{d}\int_{\Omega_{s}}\prod_{i=s+1}^{d}(X_{i}^{k}
    X_{i}^{j})dx_{i}`
    """
    #M1=M[:]
    M1=M.DiaMatrix_list[:]
    M1=[m.M for m in M1]
    M1[0],M1[i]=M1[i],M1[0]
    U1=U[:]
    U1[0],U1[i]=U1[i],U1[0]
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]
    
    Betha=1
    for j in range(1,dim):
      
        aux2=np.multiply(U1[j],R1[j])
        aux2=aux2@M1[j]
        Betha=Betha*aux2
        
    
        
    return Betha

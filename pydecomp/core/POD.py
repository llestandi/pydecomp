# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:42:18 2018

@author: Diego Britez
"""

from scipy.sparse import diags
import scipy.sparse
import numpy as np
from scipy.linalg import norm

def POD(F, Mx, Mt, tol=1e-17, rank=-1):
    """
    This function decompose a matrix as the productphi * diag(sigma) * A^t
    excuting the  POD method. Here the problem is treated as an apparent 2
    dimentions problem.
    *Parameters:*
        Mx=scipy.sparse diagonal matrix with the integration points as the
        diagonal elements for the trapezoidal method.
        Mt=scipy.sparse diagonal matrix with the integration points as the
        diagonal elements for the trapezoidal method, this matrix must have
        only one principal diagonal.
        F= Array like, Matrix with the values normaly aranged in [0]
        dimention for space and [1] dimention for time if the case.
        tol= maximal tolerance for the eigenvalues.
        rank= maiximal number of modes to be taken as long the maximal
        tolerance is not reached.
    *Returns:*

        phi= Array like, modes of the decomposition in R[0]
        sigma=array like
        A= Array like, modes of the decomposition in R[1]

    *@todo* Improve efficiency of mass matrix handling
    """
    #Verification if Mx has the scipy.sparse diagonal matrix format
    a="""
    Mx must be scipy.sparse diagonal type element, for more information read
    scipy.sparse.diags documentation.
    """
    if type(Mx)!= scipy.sparse.dia.dia_matrix:
        raise ValueError(print(a))
    b="""
    Mt must be scipy.sparse diagonal type element, for more information read
    scipy.sparse.diags documentation.
    """
    #Verification if Mt has the scipy.sparse diagonal matrix format
    if type(Mt)!= scipy.sparse.dia.dia_matrix:
        raise ValueError(print(b))
    c="""
    Mt has to be 1D dimention with 'offsets=0' scipy.sparse.diags type element,
    for more information read scipy.sparse.diags documentation.
    """

    #Verification if Mt is single diagonal sparse matrix
    if Mt.offsets!=0:
        raise ValueError(print(c))

    tshape=F.shape
    Transposed_POD=False
    if tshape[1]>tshape[0]:
        F=F.T
        Mx, Mt = Mt, Mx
        Transposed_POD=True

    C=build_correlation(F, Mx, Mt)

    Lambda , U =np.linalg.eigh(C)
    # Reversing order
    Lambda = Lambda[::-1]
    U=U[::,::-1]

    Lambda, U=truncate_POD(Lambda,tol,rank,U)
    sigma=diags(np.sqrt(Lambda))

    Mtsqinv=inv(np.sqrt(Mt))
    A=Mtsqinv.T@U
    phi=(F@Mt@A)@inv(sigma)

    if Transposed_POD:
        A,phi=phi,A

    return phi, sigma, A

def build_correlation(F,Mx,Mt):
    """
    This function creates the C matrix of correlation in the POD method
    """
    C=np.sqrt(Mt)@F.T@Mx@F@np.sqrt(Mt).T
    return C

def inv(Mtsq):
    """
    This function allows to  invert an diagonal matrix in scypy.sparse.diags
    format
    """
    Mtsq=Mtsq.diagonal()
    Mtsqinv=1/Mtsq
    Mtsqinv=diags(Mtsqinv)
    return Mtsqinv

def truncate_POD(Lambda,tol,rank,U):
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

if __name__=="__main__":
    print("\n Testing POD with random matrix and Identity weights\n")
    n,m=5,6
    F=np.random.rand(n,m)
    Mx=diags(np.ones(n))
    Mt=diags(np.ones(m))
    phi, sigma, A=POD(F,Mx,Mt)
    print("phi :\n {}\n sigma:\n {} \n A:\n {}".format(phi, sigma, A))
    F_approx = phi@sigma@A.T
    err=np.linalg.norm(F_approx-F)
    print("\n Should be small : {}".format(err))

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:42:18 2018

@author: Diego Britez
"""

from scipy.sparse import diags
import scipy.sparse
import numpy as np
from scipy.linalg import norm
from tensor_algebra import truncate_modes
from MassMatrices import DiaMatrix
import timeit
#import MassMatrix

def POD(F, Mx=[], Mt=[], tol=1e-10, rank=-1):
    """
    This function decompose a matrix as the productphi * diag(sigma) * A^t
    excuting the  POD method. Here the problem is treated as an apparent 2
    dimentions problem.
    *Parameters:*
        F= Array like, Matrix with the values normaly aranged in [0]
        dimention for space and [1] dimention for time if the case.
        
        Mx=numpy.ndarray, list or scipy.sparse diagonal matrix with t
        he integration points as thediagonal elements for the trapezoidal 
        method. If there is no input, solution equivalent to the SVD method 
        is going to be generated.
        Mt=ion points as thediagonal elements for the trapezoidal 
        method. If there is no input, solution equivalent to the SVD method 
        is going to be generated.
        tol= maximal tolerance for the eigenvalues.
        rank= maiximal number of modes to be taken as long the maximal
        tolerance is not reached.
    *Returns:*
        phi= Array like, modes of the decomposition in R[0]
        sigma=array like
        A= Array like, modes of the decomposition in R[1]

    *@todo* Improve efficiency of mass matrix handling
    """
    start=timeit.default_timer()
    tshape=F.shape
    #Creating mass matrices for l2 vectorial space if there is no mass matrices
    #declared
    if Mx==[]:
        Mx=np.ones(tshape[0])
    if Mt==[]:
        Mt=np.ones(tshape[1])
    mx=DiaMatrix(Mx)
    mt=DiaMatrix(Mt)
   
    Transposed_POD=False
    if tshape[1]>tshape[0]:
        F=F.T
        mx.M, mt.M = mt.M, mx.M
        Transposed_POD=True 
    C=build_correlation(F, mx, mt)
    Lambda , U =np.linalg.eigh(C)
    # Reversing order
    Lambda = Lambda[::-1]
    U=U[::,::-1]
    Lambda, U=truncate_modes(Lambda,tol,rank,U)
    sigma=np.sqrt(Lambda)
    
    #Mtsq is has the square root of the Mt elements
    A=(((mt.sqrt()).inv()).transpose())@U
    
    if type(mx.M)==scipy.sparse.dia.dia_matrix:
        phi=(F@mt.M@A)
        #Now we normalise phi
        #phi=the operation phi[:,np.newaxis] allows to multiply a matrix to a 
        # a vector simulating product of a matrix with a diagonal matrix
        phi=phi*(1/sigma)
        if Transposed_POD:
            A,phi=phi,A
    else:
        phi=F*mt.M@A
        phi=phi*(1/sigma)
        if Transposed_POD:
            A,phi=phi,A
    stop=timeit.default_timer()
    print(stop-start)
    return phi, sigma, A

def build_correlation(F,mx,mt):
    
    """
    This function creates the C matrix of correlation in the POD method
    """
    if type(mx.M)==scipy.sparse.dia.dia_matrix:
        C=mt.sqrt()@F.T@mx.M@F@np.sqrt(mt.M.T)
    
        
    else:
        #C=mt.sqrt()@F.T*mx.M@F*(np.sqrt(mt.M.T))
        
        C1=mt.sqrt()@F.T
        C2=mx@F
        C=C1@C2
        MTsqrt=mt.sqrt()
        Mtsqrt_transpose=MTsqrt.transpose()
        C=Mtsqrt_transpose@C.T
        C=np.transpose(C)
       
        
        #C=C*(np.sqrt(mt.M.T))
    return C

"""
if __name__=="__main__":
    print("\n Testing POD with random matrix and Identity weights\n")
    n,m=5,6
    F=np.random.rand(n,m)
    Mx=diags(np.ones(n))
    Mt=diags(np.ones(m))
    phi, sigma, A=POD(F,Mx,Mt)
    print("phi :\n {}\n sigma:\n {} \n A:\n {}".format(phi, sigma, A))
    F_approx = phi*sigma@A.T
    err=np.linalg.norm(F_approx-F)
    print("\n Should be small : {}".format(err))
"""
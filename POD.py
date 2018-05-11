# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:42:18 2018

@author: Diego Britez
"""

from scipy.sparse import diags
import scipy.sparse
import numpy as np


def POD2(F, Mx, Mt, tol=1e-17, rank=-1):
    
    """
    This function factors the matrix as phi * diag(sigma) * A^t  excuting the  
    POD method. Here the problem is treated as an apparent 2 dimentions 
    problem. 
    Parameters:
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
    Returns:
        
        phi= Array like, modes of the decomposition in R[0]
        sigma=array like
        A= Array like, modes of the decomposition in R[1]
        
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
    aux=0
    if tshape[1]>tshape[0]:
            F=F.T
            Mx, Mt = Mt, Mx
            aux=1
    
    C=build_correlation(F, Mx, Mt)
    
    Lambda , U =np.linalg.eigh(C)
    #To order the values from higher to lower in lambda vector
    Lambda = Lambda[::-1]
    #To order the values from higher to lower in U matrix
    U=U[::,::-1]
    
    Lambda, U=test_lambda(Lambda,tol,rank,U)
    
    sigma=diags(np.sqrt(Lambda))
    
    Mtsq=np.sqrt(Mt)
    Mtsqinv=inv(Mtsq)
    A=Mtsqinv.T@U
    phi=F@Mt@A
    phi=phi@inv(sigma)
    
    
    if aux==1:
        A,phi=phi,A
        
    return phi, sigma, A
#------------------------------------------------------------------------------

#Auxiliar functions used in POD2    


        
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

def test_lambda(Lambda,tol,rank,U):
    """
    This function evaluates the values of eingenvalues comparing to the maximal
    tolerance or the maximal number of rank(modes) in order to avoid nan values 
    and unnecesary calcul, the final number of modes will be reduced.
    """
    if rank==-1:
        rank=np.size(Lambda)
    i=0
    lam=[]
    
    while (Lambda[i]>tol):
        lam.append(Lambda[i])
        i+=1
        if i==rank:
            break           
    Lambda=lam
    U=U[::,:i]
    return Lambda, U
        
    
    
    



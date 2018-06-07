# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:47:32 2018

@author: Diego Britez
"""

import numpy as np

"""In this module many variables are going to be calculated such as Gammax, Gammay, Betax, Betay"""

def alpha(R,M, i, dim):
    
    """This function calculates the value of the 
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
    M1=M[:]
    M1[0],M1[i]=M1[i],M1[0]
   
    for j in range(dim-1,0,-1):
            
            R1[j]=np.multiply(R1[j],R1[j])
            aux=R1[j]@M1[j]
            alpha=alpha*aux
    return alpha
    

def gamma(R,F,M,i,dim):
    """
    This function will return the value of the 
    :math:`gamma
    variable for each iteration in the fix poitn algorithm.
    :math:`\gamma^{s}(x_{s})= \int_{\Omega/\Omega_{s}}(\prod_{i=1}^{s-1}
    X_{i}^{k+1}\prod_{i=s+1}^{D}X_{i}^{k}).F`
    """
    F2=F
    F2=np.swapaxes(F2,0,i)        
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]
    M1=M[:]
    M1[0],M1[i]=M1[i],M1[0]
    
    
    for j in range(dim-1,0,-1):
        
        F2=np.multiply(F2,R1[j])
        F2=F2@M1[j]
    return F2



def betha(M,R,U,i,dim):
    
     """
     This function calculates the value of the 
     :math:`\\beta`
     variable in the fix point algorithm:  \n
     
     :math:`\\beta^{s}(j)=\int_{\Omega/\Omega_{s}}\prod_{i=1}^{s-1}(X_{i}^{k+1}
     X_{i}^{j})\prod_{i=s+1}^{d}(X_{i}^{k}X_{i}^{j})`, which  could be expressed
     as, \n
     :math:`\\beta^{s}(j)=\prod_{i=1}^{s-1}\int_{\Omega}(X_{i}^{k+1}X_{i}^{j})
     \prod_{i=s+1}^{d}\int_{\Omega_{s}}\prod_{i=s+1}^{d}(X_{i}^{k}
     X_{i}^{j})dx_{i}`
     """
     M1=M[:]
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
     
""" 
if __name__=="__main__":
    F=np.arange(120)
    F=F.reshape(5,6,4)
    lower_limit=np.array([0,0,0])
    upper_limit =np.array([1,1,1])
    tshape=np.array([5,6,4])
    dim=3

    Vector = CartesianGrid.CartesianGrid(dim, tshape, lower_limit, upper_limit)

    X = Vector.SpaceCreator()    
    New_Solution=IterationSolution(tshape,dim)
    New_Solution._current_solution_initialization()
    R=New_Solution.R
    i=1

    F2=F
    R2=R
    R2[0],R2[i]=R2[i],R2[0]
    X2=X
    X2[0],X2[i]=X2[i],X2[0]
    F2=np.swapaxes(F2,0,i)


    for j in range(dim-1,1,-1):
        F2=np.multiply(F2,R2[j])
        F2=integration_1dtrap(F2,X2[j]) 
"""


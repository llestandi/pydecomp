# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:48:14 2018

@author: Diego Britez
"""

import numpy as np
from scipy.linalg import norm
from variables import gamma, betha, alpha
from Iteration_solution import IterationSolution

            
    
def fixpoint(X,tshape,U,F,z,r,maxfix):
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
    **Returns**
    R = n mode of the decomposition in the PGD decomposition method.\n
    z = actualized number of iteration that did not converge, so: \n
    :math:`z_{n}=z_{n-1}+1` if the actual mode does not converge or \n
    :math:`z_{n}=z_{n-1}`if it had converged.
    """
    dim=np.size(tshape)
    New_Solution=IterationSolution(tshape,dim)
    New_Solution._current_solution_initialization()
    R=New_Solution.R
    
    
    Old_Solution=New_Solution
    k=0                                    
    eppf=1e-8
    epsilon=1              
    itmax=maxfix
    
    #Eliminate the commentary if the impresion in screen about the 
    #the information of convergence at each iteration is desired
    """
    if r==1: 
        
        o='''Number of iterations in fix point routine for each enrichment'''
        g='''-------------------------------------------------------------'''
        print(o)
        print(g)
    """
    while ((epsilon>=eppf) & (k<itmax)): 
       
        k=k+1
        
        Old_Solution=R[dim-1]
        Gamma=[]
        Alpha=np.zeros(dim)
        
        for i in range(dim):
            
            Alpha=alpha(R,X,i,dim)
            Gamma=gamma(R,F,X,i,dim)
            Betha=betha(X,R,U,i,dim)
            
            aux=np.dot(np.swapaxes(U[i],0,1),Betha)
            aux=np.transpose(aux)
            R[i]=(-aux+Gamma)/Alpha
            
            if (i<(dim-1)):
               
                R[i]=R[i]/(norm(R[i]))
                
        epsilon=norm(R[dim-1]-Old_Solution)/norm(Old_Solution)
        
        
        #Eliminate the comentary if the impresion in screen about the 
        #the information of convergence at each iteration is desired
        
        
        """   
        if epsilon<eppf :
            print('k(',r,')=',k)
            print('convergence')
            
        if k==itmax:
            print('k(',r,')=',k)
            print('No convergence')
        """
        if k==itmax:
            z=z+1
            
        
    return  R,z




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
    This function calculates the solution mode for each iteration in the 
    enrichment loop. The definition of each variable (such as alpha, betha,  
    etc). that it's used herecould be found in detail in the manuscript of 
    Lucas Lestandi doctoral thesis.
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
            
        if epsilon<eppf :
            print('k(',r,')=',k)
            print('convergence')
            """
            print('R0',R[0])
            print('shapeR0',np.shape(R[0]))
            """
        if k==itmax:
            print('k(',r,')=',k)
            print('No convergence')
            z=z+1
            
        
    return  R,z




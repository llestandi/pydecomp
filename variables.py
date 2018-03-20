# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:47:32 2018

@author: Diego Britez
"""

import numpy as np
from integration import integration_1dtrap
"""In this module many variables are going to be calculated such as Gammax, Gammay, Betax, Betay"""

def alpha(R,X, i, dim):
    """Cette fonction calcule la valeur du module alpha a partir du vecteur S ou R 
    en étude et son chap de domaine X ou Y"""
    R1=R[:]
    alpha=1
    if (i>0):
        for j in range(i):
            R1[j]=np.multiply(R1[j],R1[j])    
            aux=integration_1dtrap(R1[j],X[j])
            alpha=alpha*aux
            
        
    if (i<dim-1):
        for j in range(i+1,dim):
            R1[j]=np.multiply(R1[j],R1[j])   
            aux=integration_1dtrap(R1[j],X[j])
            alpha=alpha*aux
       
   
    return alpha
    

def gamma(R,F,X,i,dim):
    F2=F
    F2=np.swapaxes(F2,0,i)        
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]
    X1=X[:]
    X1[0],X1[i]=X1[i],X1[0]
    
    
    for j in range(dim-1,0,-1):
        
        F2=np.multiply(F2,R1[j])
        
        F2=integration_1dtrap(F2,X1[j])  
  
    return F2



def betha(X,R,U,i,dim):
     
     X1=X[:]
     X1[0],X1[i]=X1[i],X1[0]  
     U1=U[:]
     U1[0],U1[i]=U1[i],U1[0]
     R1=R[:]    
     R1[0],R1[i]=R1[i],R1[0]
     
     
     Betha=1
     for j in range(1,dim):
         
         aux2=np.multiply(U1[j],R1[j])
         aux2=integration_1dtrap(aux2,X1[j])
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

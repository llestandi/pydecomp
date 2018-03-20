# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:48:14 2018

@author: Diego Britez
"""

import numpy as np
from scipy.linalg import norm
from variables import gamma, betha, alpha
from Iteration_solution import IterationSolution

            
    
def fixpoint(X,tshape,U,F,z,r):
    dim=np.size(tshape)
    New_Solution=IterationSolution(tshape,dim)
    New_Solution._current_solution_initialization()
    R=New_Solution.R

    
    
    Old_Solution=New_Solution
    k=0                                           #initialization de la fonction du point fixe
    eppf=1*10**(-10)
    epsilon=1                                     #Première valeure de epsilon pour comencer l'algorithme
    itmax=30
    
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
    
"""
Gammax=gammax(F,S,nx,ny,Y)
        Bethax=betax(SS,S,Y)
        #print('shape bethax', np.shape(Bethax))
        #print(Bethax)
        #print('shape RR', np.shape(RR))
        #print(RR)
        aux=np.dot(np.transpose(RR),Bethax)
        aux=np.transpose(aux)
        R=(-aux+Gammax)/Alphax      
        R=R/(norm(R))
      
        
        #Partie 2
        Alphay=alpha(R,X)
        Gammay=gammay(F,R,nx,ny,X)
        Bethay=betay(RR,R,X)
        aux=np.dot(np.transpose(SS),Bethay)
        aux=np.transpose(aux)
        S=(-aux+Gammay)/(Alphay)
        epsilon=norm(S-Sv)/norm(Sv)
"""


    



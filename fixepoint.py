# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:48:14 2018

@author: Diego Britez
"""

import numpy as np
from scipy.linalg import norm
from variables import gammax, gammay, betax, betay, alpha


def fixpoint(X,Y,SS,RR,nx,ny,F,z):
    S=np.array([np.ones(Y.size)])                 #Creation du vecteur S0
    R=np.array([np.zeros(X.size)])                #Creation du vecteur R
    Sv=S
    k=0                                           #initialization de la fonction du point fixe
    eppf=1*10**(-10)
    epsilon=1                                     #Première valeure de epsilon pour comencer l'algorithme
    itmax=30
    while ((epsilon>=eppf) & (k<itmax)): 
        k=k+1
        print('k',k)
        Sv=S
        Alphax=alpha(S,Y)
        Gammax=gammax(F,S,nx,ny,Y)
        Bethax=betax(SS,S,Y)
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
       
        if epsilon<eppf :
            print('convergence')
        if k==itmax:
            z=z+1
            
        
    return  S,R,z
    



    



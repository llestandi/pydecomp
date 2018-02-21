# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:47:32 2018

@author: Diego Britez
"""

import numpy as np
from integration import integration_1dtrap
"""In this module many variables are going to be calculated such as Gammax, Gammay, Betax, Betay"""

def alpha(phi,domaine):
    """Cette fonction calcule la valeur du module alpha a partir du vecteur S ou R 
    en étude et son chap de domaine X ou Y"""
    phi=phi**2
    alpha=integration_1dtrap(phi,domaine)
    return alpha
    
    """Cette fonction calcule la fonction Gammax décrit en la bibliographie (Manuscrite de Lucas
    à la page 22)
        Il existe une difèrence en la nomenclature, ici la fonction Yk est appellée S et on continue 
    avec cette nomenclature pendant tout la programation.
        Pour obtenir la valeure correct à la multiplication entre S et la matrice F"""

def gammax(F,S,nx,ny,Y):
    F2=np.multiply(S,np.transpose(F))
    Gammax=integration_1dtrap(F2,Y)
    """
    #Autre méthode mais moins efficace
    
    F2=np.transpose(F2)
    I=np.zeros(ny)
    Var=0
    Gammax=[0]
    for i in range (nx):
        I[:]=F2[:,i]
        Var=integration_1dtrap(I,Y)
        Gammax=np.append(Gammax,Var)
    Gammax=Gammax[1:]
    """    
    return np.array([Gammax])

def gammay(F,R,nx,ny,X):
    F2=np.multiply(R,F)
    Gammay=integration_1dtrap(F2,X)
    
    """
    #Autre méthode mais moins efficace
    I=np.zeros(nx)
    Var=0
    Gammay=[0]
    for i in range (ny):
        I[:]=F2[i,:]
        Var=integration_1dtrap(I,X)
        Gammay=np.append(Gammay,Var)
    Gammay=Gammay[1:]
    """
    return np.array([Gammay])

def betax(SS,S,Y):
   
    SSS=np.multiply(SS,S)
    Betax=integration_1dtrap(SSS,Y)
    
    """
    #Autre méthode mais moins efficace
    a=np.shape(SS)
    j=a[0]                                       
    Betax=np.zeros(j)
    for i in range (j):
       Betax[i]=integration_1dtrap(SSS[i,:],Y)
    """  
    return Betax

def betay(RR,R,X):
    
    RRR=np.multiply(RR,R)
    Betay=integration_1dtrap(RRR,X)
    """
    #Autre méthode mais moins efficace
    a=np.shape(RR)
    j=a[0]
    Betay=np.zeros(j)
    for i in range (j):
        Betay[i]=integration_1dtrap(RRR[i,:],X)
    """
    return Betay
    


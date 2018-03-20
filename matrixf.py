# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:16:03 2018

@author: Diego Britez
"""

import numpy as np

"""This function will return the values of a continous function pfunc(x,y) 
in to a discretised domaine of a matrix, the input values are the vectors X and Y
of the domaine, the function pfunc is required inside this application"""

def pfunc(x,y): 
    return 1/(1+(x*np.e**(y)))
    #Fonctions tets à utiliser 
    # x*y                              #fonction test 1
    # 1/(1+(x*y))                      #fonction test 2
    # np.sin(np.sqrt(x**2+y**2))       #fonction test 3
    # np.sqrt(1-x*y)                   #fonction test 4
    # 1/(1+(x*np.e**(y)))              #fonction test 5

def matfunc(X,tshape):    
    
    XX,YY=np.meshgrid(X[0],X[1])
    Function=np.reshape(pfunc(XX,YY),(tshape[1],tshape[0]))
    Function=np.transpose(Function)
    return Function

if __name__=="__main__":
    X=[]
    x1=np.linspace(0,1,3)
    x2=np.linspace(0,1,5)
    X.append(x1)
    X.append(x2)
    tshape=np.array([3,5])
    F=matfunc(X,tshape)
    print(F)





"""   
def matfunc(X,div_tshape):
    nx=np.size(X)
    ny=np.size(Y)

    G,P=np.meshgrid(X,Y)
    GG=np.reshape(G,(nx*ny))
    PP=np.reshape(P,(nx*ny))
    Function=np.zeros((nx*ny))
    Function[:]=pfunc(GG[:],PP[:])
    Function=np.reshape(Function,(ny,nx))
    return Function
"""    
    
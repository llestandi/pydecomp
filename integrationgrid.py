# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:47:48 2018

@author: Diego Britez
"""

import numpy as np
from grille import grille
"""
In this class we define the grid that is going to be needed to integrate with
the Method of the trapezes
""" 
class IntegrationGrid:
   
    def __init__(self,X,dim, tshape):
        self.X = X
        self.Xgrid = []
        self.dim = dim
        self.tshape = tshape
    
    def IntegrationGridCreation(self):
        for i in range(self.dim):
            w=grille(self.X[i],self.tshape[i])
            self.Xgrid.append(w)
        return self.Xgrid    
    
if __name__=="__main__":
    X=[]
    x1=np.linspace(0,1,3)
    x2=np.linspace(0,1,5)
    X.append(x1)
    X.append(x2)
    tshape=np.array([3,5])
    Xgrid=IntegrationGrid(X,np.size(tshape),tshape)
    Xgrid=Xgrid.IntegrationGridCreation()        
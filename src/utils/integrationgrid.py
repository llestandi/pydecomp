# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:47:48 2018

@author: Diego Britez
"""

import numpy as np

"""
In this class we define the grid that will contain the integration points to 
integrate with the trapezes  Method.
""" 
class IntegrationGrid:
   
    def __init__(self,X,dim, tshape):
        self.X = X
        self.Xgrid = []
        self.dim = dim
        self.tshape = tshape
        
    def grille(self,x,nx):
        w=np.zeros(nx)
        w[1:-1]=(x[2:]-x[0:-2])/2
        w[0]=(x[1]-x[0])/2
        w[-1]=(x[-1]-x[-2])/2
        return w
    
    def IntegrationGridCreation(self):
        for i in range(self.dim):
            w=self.grille(self.X[i],self.tshape[i])
            self.Xgrid.append(w)
        return self.Xgrid    
"""    
if __name__=="__main__":
    #X=[]
    #x1=np.linspace(0,1,3)
    #x2=np.linspace(0,1,5)
    #X.append(x1)
    #X.append(x2)
    #tshape=np.array([3,5])
    Xgrid=IntegrationGrid(X,np.size(tshape),tshape)
    Xgrid=Xgrid.IntegrationGridCreation()
    print(Xgrid)
"""        
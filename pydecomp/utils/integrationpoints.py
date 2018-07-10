#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 15:22:54 2018

@author: diego
"""
import numpy as np
"""
In this class we define the grid that will contain the integration points to
integrate with the trapezes  Method.
"""
class IntegrationPoints:

    def __init__(self,X,dim, tshape):
        self.X = X
        self.Xgrid = []
        self.dim = dim
        self.tshape = tshape

    def trapeze(self,x,nx):
       
        w=np.zeros(nx)
        w[1:-1]=(x[2:]-x[0:-2])/2
        w[0]=(x[1]-x[0])/2
        w[-1]=(x[-1]-x[-2])/2
        return w

    def IntegrationPointsCreation(self):
        for i in range(self.dim):
            w=self.trapeze(self.X[i],self.tshape[i])
            self.Xgrid.append(w)
        return self.Xgrid
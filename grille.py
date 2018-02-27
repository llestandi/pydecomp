# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:17:15 2018

@author: Diego Britez
"""
import numpy as np
def grille(x,nx):
    w=np.zeros(nx)
    w[1:-1]=(x[2:]-x[0:-2])/2
    w[0]=(x[1]-x[0])/2
    w[-1]=(x[-1]-x[-2])/2
    return w
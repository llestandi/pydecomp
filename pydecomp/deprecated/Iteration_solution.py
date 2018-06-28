# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:42:49 2018

@author: Diego Britez
"""

import numpy as np
#from scipy.linalg import norm


class IterationSolution:
    def __init__(self,tshape,dim):
        self.R=[]
        self.tshape=tshape
        self.dim=dim
    def _current_solution_initialization(self):
        for i in range (self.dim):
            self.R.append(np.array([np.ones(self.tshape[i])]))

if __name__=="__main__":
    tshape=np.array([5,4,8])
    dim=3
    R=IterationSolution(tshape,dim)
    R._current_solution_initialization()
    R=R.R
    
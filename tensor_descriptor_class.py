# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:40:00 2018

@author: Diego Britez
"""
import numpy as np

#TensorDescriptor is a class that contains basic information that 
#other sub classes (Full Format, CannicalForme) will inherit. 
#To begin the TensorDescriptor class is defined with the 
#values of the number of grids in each space (thsape array), and 
#the number of dimentions (for the case of this code is going to be
#always equal to 2).
class TensorDescriptor:
    def __init__(self,tshape,dim):
        if (np.size(tshape)!=dim):
            print('Fatal error!!! Dimetion size declaration  ')
            print(' and the actual size of div_tshape are not equals')
            
        self._dim=dim
        self._tshape=tshape
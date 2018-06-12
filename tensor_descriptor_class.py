# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:40:00 2018

@author: Diego Britez
"""
import numpy as np

#
#To begin the TensorDescriptor class is defined with the
#values of the number of grids in each space (thsape array), and
#the number of dimentions.
class TensorDescriptor:
    """
    TensorDescriptor is a class that contains basic information that
    other  classes (Full Format, CanonicalForme) will inherit.\n
    **Attributes**
        **_tshape**: array like, with the numbers of elements that each 1-rank
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is
        going to be represented. The value of dim must be coherent with the
        size of _tshape parameter.
    """
    def __init__(self,tshape,dim):
        if (np.size(tshape)!=dim):
            raise AttributeError('Fatal error!!! Dimetion size declaration  and the actual size of div_tshape are not equals')
            
        self._dim=dim
        self._tshape=tshape

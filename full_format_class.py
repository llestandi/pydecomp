# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:42:12 2018

@author: Diego Britez
"""

import numpy as np
from tensor_descriptor_class import TensorDescriptor

        
#FullFormat is a class thats inherites the values of tshape  and the number of
#dimentions from TensorDescriptor        
class FullFormat(TensorDescriptor):
    def __init_(self,div_tshape,dim):
        TensorDescriptor.__init__(self)
        self.VAL=np.ndarray(shape=(self._tshape))
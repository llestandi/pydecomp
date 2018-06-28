# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 09:42:12 2018

@author: Diego Britez
"""

import numpy as np
from tensor_descriptor_class import TensorDescriptor

        
        
class FullFormat(TensorDescriptor):
    """
    FullFormat is a class that represents the n-dimensional array 
    numpy.ndarray.\n
    **Attributes**\n
    **_tshape**: array like, with the numbers of elements that each 1-rank 
     tensor is going to discretize each subspace of the full tensor. \n
    **dim**: integer type, number that represent the n-rank tensor that is 
    going to be represented. The value of dim must be coherent with the 
    size of _tshape parameter. \n
    **_Var:** n-dimensional array that the tensor object represents.
    
    """
    def __init__(self,_Var,_tshape,dim): 
        """Constructor for tensor object.
        dat can be numpy.array or list.
        shape can be numpy.array, list, tuple of integers"""                                                                 
        TensorDescriptor.__init__(self,_tshape,dim)
        
        if(_Var.__class__ == list):
            _Var = np.array(_Var);

        if(_tshape != None):
            if(len(_tshape) == 0):
                raise ValueError("Second argument must be a row vector.");
            
            if(_tshape.__class__ == np.ndarray):
                if(_tshape.ndim != 2 and [0].size != 1):
                    raise ValueError("Second argument must be a row vector.");
            _tshape = tuple(_tshape);
        else:
           _tshape = tuple(_Var.shape);
        
        self._tshape = _tshape;
        self._Var = _Var.reshape(self._tshape);
        self.dim=dim
    #--------------------------------------------------------------------------    
    def size(self):
        """returns the number of elements in the tensor"""
        Tam=self._Var.size
        return Tam;
    #--------------------------------------------------------------------------
    def dimsize(self, ind):
        """ returns the size of the specified dimension.
        Same as shape[ind]."""
        return self._tshape[ind]
    #--------------------------------------------------------------------------
    def ndims(self):
        """ returns the number of dimensions. """
        return len(self._tshape)
    #--------------------------------------------------------------------------
    def tondarray(self):
        """return an ndarray that contains the data of the tensor"""
        return self._Var;
        
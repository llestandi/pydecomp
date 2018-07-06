#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:46:56 2018

@author: diego
"""
import numpy as np
from scipy.sparse import diags
import scipy.sparse
class MassMatrice:
    """
    This class is created to simplify operations with mass matrices. Mass
    matrices can be defined either as a sparse matrix, as a list or as 
    a numpy.ndarray.
    This class will unify the matmul operation for any of this formats. 
    """
    possible_mass_matrix_format=[list, scipy.sparse.dia.dia_matrix,
                                 numpy.ndarray]
    list_or_array=[list, numpy.ndarray]
    def __init__(self,M):
        self.M=M
    
    def __matmul__(self,Matrix):
        if type(self.M)==scipy.sparse.dia.dia_matrix:
            result=self.M@Matrix
        elif (type(self.M)==numpy.ndarray):     
            self.M=self.M.reshape([1,int(self.M.shape[0])])
            result=np.multiply(Matrix.T,self.M)
            result=result.T
        elif type(self.M)==list:
            M1=np.array(self.M)
            M1=M1.reshape([1,int(M1.shape[0])])
            result=np.multiply(Matrix.T,M1)
            result=result.T
        return result
    
    def inv(self):
        """
        Thsi fonction will return the inverse of mass matrix.
        """
        possible_mass_matrix_format=[list, scipy.sparse.dia.dia_matrix,
                                 numpy.ndarray]
        
        list_or_array=[list, numpy.ndarray]
        
        if type(self.M)==scipy.sparse.dia.dia_matrix:
            Maux=self.M
            Maux=Maux.diagonal()
            inv=1/Maux
            inv=diags(inv)      
        if type(self.M) in list_or_array:
            inv=1/self.M
        return inv
    
        
            
        
        
        
        
        
        


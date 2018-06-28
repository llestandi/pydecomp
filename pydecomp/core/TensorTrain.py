#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:20:10 2018

@author: diego
"""
import numpy as np
from deprecated.tensor_descriptor_class import TensorDescriptor
import csv
import pickle
class TensorTrain:
    """  
          
    **Tensor Train Type Format**
    In this format, any tensor 
    :math:`\chi\in V = \otimes_{\mu=1}^{d}V_{\mu}`
    a tensor space, is written as the finite sum of rank-1 tensors.
    :math:`\chi \in C_{r} (\mathbb{R})`
    is said to be represented in the 
    Tensor Train format and it reads, \n
    :math:`\chi(i_{1},...,i_{d})=\sum\limits_{\alpha_{0},...,\alpha_{d-1},
    \alpha_{d}}G_{1}(\alpha_{0},i_{1},\alpha_{1})G_{2}(\alpha_{1},i_{2},
    \alpha_{2})...G_{d}(\alpha_{d-1},i_{d},\alpha_{d})`
    \n
    **Attributes**
    
        **_G** : list type, each element will bi a 3 dimentional tensor, one 
        element (tensor) for each dimention to be decomposed.
        **_tshape**: array like, with the numbers of elements that each 1-rank 
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is 
        going to be represented. The value of dim must be coherent with the 
        size of _tshape parameter. \n
        **_ranks**: list type, this element will contain the number of rank
        obtained in each decomposition. 
    **Methods**
    """
    
    def __init__(self,G):  
        self.G=G 
        for i in range(len(G)):
            if len(G[i].shape)!=3:
                 raise ValueError('All elements must be 3 dimentional ndarray')                                                                  
        self.ranks=[]
        self.G=G[:]
        #loading ranks element
        for i in range(len(self.G)):
            actual_rank=self.G[i].shape[2]
            self.ranks.append(actual_rank)
        #Loading dim element
        self.dim=len(self.G)

        
        
    def reconstruction(self):
        """
        This method will transform a TensorTrain object in to a numpy Tensor
        element.\n
        **Parameters**\n
        G=TensorTrain format element.\n
        **Returns**\n
        Reconstructor= numpy nd array. 
        
        """
        GG=self.G[:]
        ranks=self.ranks
        dim=len(GG)
    
        tshape=[]
        for i in range(dim):
            tshape.append(GG[i].shape[1])
        
        Reconstructor=GG[0].reshape([tshape[0],ranks[0]])
        for i in range(1,dim-1):
            
            GG[i]=GG[i].reshape([ranks[i-1],tshape[i]*ranks[i]])
            Reconstructor=Reconstructor@GG[i]
            size_reconstructor=Reconstructor.size
            Reconstructor=Reconstructor.reshape([int(size_reconstructor/ranks[i]),ranks[i]])
        
        GG[dim-1]=GG[dim-1].reshape([ranks[dim-2],tshape[dim-1]])
        Reconstructor=Reconstructor@GG[dim-1]
        Reconstructor=Reconstructor.reshape(tshape)
    
        return Reconstructor  
        
    
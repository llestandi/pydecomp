#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 15:54:55 2018

@author: diego
"""

import numpy as np
from TSVD import TSVD
from TensorTrain import TensorTrain

def TT_SVD(F, eps=1e-10, rank=100):
    """
    Returns the decomposed form of a Tensor in the Tensor Train format by 
    using the TT-SVD decomposition method described by I. V. Oseledets. \n 
    **Parameters** \n
    F= numpy.ndarray type. N dimentional tensor. \n
    eps= maximal error expected in each decomposition. \n
    rank= integer type . Maximal number of rank in each decomposed linear 
    network.\n
    **Returns** \n
    G= Tensor Train class element. \n
    
    **Exemple** \n

    F=np.arange(720) \n

    F=F.reshape([8,9,10]) \n

    G=TT_SVD(F, rank=5) \n
    
    
    """
    tshape=F.shape
    
    dim=len(tshape)
    r=[1]
    C=F
    G=[]
    for i in range(dim-1):
       aux=r[i]*tshape[i]
       Csize=C.size
       C=C.reshape(aux,int(Csize/aux))
       
       if rank!=100:
           u,sigma,v=TSVD(C, rank=rank)
       else:
           u,sigma,v=TSVD(C, epsilon=eps)
       
       new_rank=sigma.shape[0]
       r.append(new_rank)
       G.append(u.reshape(r[i],tshape[i],r[i+1]))
       C=sigma@v.T
    G.append(C)
    
    Gdshape=list(G[dim-1].shape)
    Gdshape.append(1)
    
    G[dim-1]=G[dim-1].reshape(Gdshape)
    return TensorTrain(G)


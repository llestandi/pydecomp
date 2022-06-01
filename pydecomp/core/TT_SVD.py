#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 13:19:41 2018

@author: diego
"""
import numpy as np
from pydecomp.core.TSVD import TSVD
import pydecomp.core.MassMatrices as mm
from pydecomp.core.POD import POD
from pydecomp.core.TensorTrain import  TT_init_from_decomp
from copy import deepcopy
from math import floor
from time import time

def TT_SVD(F, eps=1e-8, rank=-1, MM=None,solver='EVD',QTTcutoff=1,verbose=0):
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
    if MM:
        M=deepcopy(MM)
        is_first=True

    for i in range(dim-1):
        Cshape=(r[i]*tshape[i],C.size//(r[i]*tshape[i]))
        C=np.reshape(C,Cshape)
        if verbose>0:
            print("Dim {} of {}".format(i,dim-1))
            start=time()
        if not MM:
            if min(Cshape)>1000:
                #trick for cutting very large ranks in QTT (enbable SVD to be computed in a reasonable time)
                rank=floor(min(Cshape)*QTTcutoff)
            u,sigma,v=TSVD(C,epsilon=eps, rank=rank, solver=solver)
        else:
            Mx,Mt = mm.matricize_mass_matrix_for_TT(M,is_first)
            u,sigma,v=POD(C, Mx, Mt, tol=eps, rank=rank)

        new_rank=sigma.shape[0]
        r.append(new_rank)
        G.append(u.reshape(r[i],tshape[i],r[i+1]))
        C =(sigma*v).T

        if MM:
            if i>0:
                M=mm.pop_1_MM(M)
            M.update_mass_matrix(0, mm.identity_mass_matrix(C.shape[0],M.is_sparse))
            is_first=False
            
        if verbose>0:
            start=time()
            print("Walltime : {:.2f}s".format(time()-start))

    G.append(C)
    Gdshape=list(G[dim-1].shape)
    Gdshape.append(1)
    G[dim-1]=G[dim-1].reshape(Gdshape)
    return TT_init_from_decomp(G)

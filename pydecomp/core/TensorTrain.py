#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:20:10 2018

@author: diego
"""
import numpy as np
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

        **G** : list type, each element will bi a 3 dimentional tensor, one
        element (tensor) for each dimention to be decomposed.
        **shape**: array like, with the numbers of elements that each 1-rank
        tensor is going to discretize each subspace of the full tensor. \n
        **ndim**: integer type, number that represent the n-rank tensor that is
        going to be represented. The value of dim must be coherent with the
        size of _tshape parameter. \n
        **rank**: list type, this element will contain the number of rank
        obtained in each decomposition. Includes r0 and rd for easier loops.
    **Methods**
    """

    def __init__(self,shape):
        self.G=[]
        self.rank=[]
        self.ndim=len(shape)
        self.shape=shape

    def fill(self,G):
        """ Fills a tensor from list G """
        if self.ndim != len(G):
            raise AttributeError("G has incorrect length \
                                 {} != {}".format(len(G),self.ndim))
        for i in range(len(G)):
            if len(G[i].shape)!=3:
                raise ValueError('All elements must be 3 dimentional ndarray')
            if G[i].shape[1] != self.shape[i]:
                raise ValueError("Shape [{}] does not match".format(G[i].shape[1]))
        self.G=G
        #Setting ranks element
        self.rank=[1]
        for i in range(len(self.G)):
            r=self.G[i].shape[2]
            self.rank.append(r)
        if self.rank[-1] !=1:
            raise AssertionError("Last rank should be 1, {}".format(self.rank))
        return
    def __str__(self):
        str="Tensor Train\n"
        str+="ndim={}\n".format(self.ndim)
        str+="shape={}\n".format(self.shape)
        str+="rank={}\n".format(self.rank)
        for i in range(self.ndim):
            str+="\n G[{}]: \n {}".format(i,G[i])
        return str

    def to_full(self,trunc_rank=[]):
        """
        This method will transform a TensorTrain object in to a numpy Tensor
        element.\n
        **Parameters**\n
        G=TensorTrain format element.\n
        **Returns**\n
        Reconstructor= numpy nd array.

        """
        G=self.G[:]
        if trunc_rank==[] or not (trunc_rank<=self.rank):
            r=self.rank
        else:
            r=trunc_rank
        dim=self.ndim
        shape=self.shape

        full=G[0][:,:,:r[1]].reshape([shape[0],r[1]])
        for i in range(1,dim):
            GG=np.reshape(G[i][:r[i],:,:r[i+1]],[r[i],shape[i]*r[i+1]])
            full=full@GG
            size=full.size
            full=full.reshape([int(size/r[i+1]),r[i+1]])

        return full.reshape(shape)

    def trunc(self,rank):
        """Truncation of Tensor Train at rank"""
        raise NotImplementedError("Truncation function not implemented yet!")

def init_from_decomp(G):
    """ Initializes and fills a tensor train TT from data obtained in TTSVD
    from list of order 3 tensors G """
    ndim=len(G)
    shape=np.asarray([G[i].shape[1] for i in range(ndim)])
    TT=TensorTrain(shape)
    TT.fill(G)
    return TT

if __name__=="__main__":
    d=3
    shape=[10,11,12]
    r=[1,2,3,1]
    G=[np.random.rand(r[i],shape[i],r[i+1]) for i in range(d)]
    print([g.size for g in G])
    TT=init_from_decomp(G)
    print(TT)
    print(np.linalg.norm(TT.to_full()-TT.to_full([1,2,3,1])))

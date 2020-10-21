#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:20:10 2018

@author: diego
"""
import numpy as np
from core.tensor_algebra import norm
from utils.misc import rank_sampling


class TensorTrain:
    """

    **Tensor Train Type Format**
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
            str+="\n G[{}]: \n {}".format(i,self.G[i])
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
        r=self.check_rank(trunc_rank)
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

    def mem_eval(self,rank):
        """ Evaluates memory use of self at given rank (list of integer)"""
        mem_use=0
        r=self.check_rank(rank)
        for i in range(self.ndim):
            mem_use+=r[i]*self.shape[i]*r[i+1]
        return mem_use

    def check_rank(self,rank=[]):
        """ Verifies that rank is acceptable for self. If not, returns the actual rank"""
        zeros=[0 for i in range(self.ndim+1)  ]
        if np.all(rank<=self.rank) and np.all(rank>zeros):
            r=rank
        else:
            r=self.rank
        r=[int(rk) for rk in r]
        return r

def TT_init_from_decomp(G):
    """ Initializes and fills a tensor train TT from data obtained in TTSVD
    from list of order 3 tensors G """
    ndim=len(G)
    shape=np.asarray([G[i].shape[1] for i in range(ndim)])
    TT=TensorTrain(shape)
    TT.fill(G)
    return TT

def error_TT_data(T_tt,T_full, M=None,Norm="L2",sampling="default"):
    """
    @author : Lucas 27/06/18
    Builds a set of approximation error and associated compression rate for a
    representative subset of ranks.

    **Parameters**:
    *TT*     TensorTrain approximation
    *T_full* Full tensor

    **Return** ndarray of error values, ndarray of compression rates

    **Todo** Add integration matrices to compute actual error associated with
    discrete integration operator
    """
    if np.any(T_full.shape != T_tt.shape):
        raise AttributeError("T_full (shape={}) and TT (shape={}) should have \
                             the same shape".format(T_full.shape,T_tt.shape))
    #We are going to calculate one average value of ranks
    d=T_full.ndim
    data_compression=[]
    shape=T_full.shape
    F_volume=np.product(shape)
    rank=np.asarray(T_tt.rank)
    maxrank=max(rank)
    print("computing erorr with maxrank={}".format(maxrank))
    error=[]
    comp_rate=[]
    norm_T=norm(T_full,M,Norm)

    r=np.zeros(d+1)
    for i in rank_sampling(maxrank,sampling):
        r=np.minimum(rank,i)
        comp_rate.append(T_tt.mem_eval(r)/F_volume)
        T_approx=T_tt.to_full(r)
        actual_error=norm(T_full-T_approx, M,Norm)/norm_T
        error.append(actual_error)

    return np.asarray(error), np.asarray(comp_rate)

def error_TT_data_complete(T_tt,T_full, M=None,sampling="exponential_fine"):
    """
    @author : Lucas 27/06/18
    Builds a set of approximation error and associated compression rate for a
    representative subset of ranks.

    **Parameters**:
    *TT*     TensorTrain approximation
    *T_full* Full tensor

    **Return** ndarray of error values, ndarray of compression rates

    **Todo** Add integration matrices to compute actual error associated with
    discrete integration operator
    """
    if np.any(T_full.shape != T_tt.shape):
        raise AttributeError("T_full (shape={}) and TT (shape={}) should have \
                             the same shape".format(T_full.shape,T_tt.shape))
    #We are going to calculate one average value of ranks
    print("Computing approximation error chart of TT decomposition with {} sampling".format(sampling))
    d=T_full.ndim
    data_compression=[]
    shape=T_full.shape
    F_volume=np.product(shape)
    rank=np.asarray(T_tt.rank)
    maxrank=max(rank)
    error=[]
    comp_rate=[]
    norm_T={"L1":norm(T_full,M,type="L1"),
            "L2":norm(T_full,M,type="L2"),
            "Linf":norm(T_full,M,type="Linf")}

    r=np.zeros(d+1)
    error={"L1":[],"L2":[],"Linf":[]}
    for i in rank_sampling(maxrank,sampling):
        r=np.minimum(rank,i)
        print(r)
        comp_rate.append(T_tt.mem_eval(r)/F_volume)
        T_approx=T_tt.to_full(r)
        error["L1"].append(norm(T_full-T_approx,M,type="L1")/norm_T["L1"])
        error["L2"].append(norm(T_full-T_approx,M,type="L2")/norm_T["L2"])
        error["Linf"].append(norm(T_full-T_approx,M,type="Linf")/norm_T["Linf"])

    return error, np.asarray(comp_rate)

if __name__=="__main__":
    d=3
    shape=[10,11,12]
    r=[1,2,3,1]
    G=[np.random.rand(r[i],shape[i],r[i+1]) for i in range(d)]
    print([g.size for g in G])
    TT=TT_init_from_decomp(G)
    print(TT)
    print(np.linalg.norm(TT.to_full()-TT.to_full([1,2,3,1])))
    print("Error eval:",error_TT_data(TT,TT.to_full()))

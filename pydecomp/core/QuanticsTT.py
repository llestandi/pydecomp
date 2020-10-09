#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Oct 8 12:09:16 2020

@author: Lucas

Wrapping TT operation on a quantized tensor/vector
"""


import numpy as np
import math
import core.TensorTrain as TT
from core.TT_SVD import TT_SVD

class QuanticsTensor:
    """
    **Quantics Tensor Train Type Format**
    **Attributes**
    **Methods**
    """

    def __init__(self,data):
        self.original_shape=data.shape
        self.qshape=[]
        self.q=None
        self.data=data # can be either a numpy array (init) or pointing to a TensorTrain object (pointer, no copy)
        self.approx_data=None # can be either a numpy array (init) or pointing to a TensorTrain object
        self.approx_error=None
        self.size=data.size
        self.ndim=len(data.shape)

    def __str__(self):
        str="QuanticsTT\n"
        str+="ndim={}\n".format(self.ndim)
        str+="original_shape={}\n".format(self.original_shape)
        str+="q={}\n".format(self.q)
        str+="qshape={}\n".format(self.qshape)
        # str+="data=\n{}\n".format(self.data)
        str+="-------------------------------------------\n"
        str+="Approx_data rank=\n{}\n".format(self.approx_data.rank)
        return str

    def reshape_to_q(self,q):
        #testing size and q fit
        D=math.log( self.size, q )
        if not D.is_integer() :
            raise ValueError("q cannot quantize size")
        self.q=q
        self.qshape=[q] * int(D)
        print("reshaping to {}".format(self.qshape))
        self.data=np.reshape(self.data,self.qshape)

        return

    def reshape_manual(self,qshape):
        raise NotImplementedError("not programmed yet")
        return

    def applyTTSVD(self,eps=1e-3,rank=-1,MM=None):
        self.approx_data=TT_SVD(self.data, eps, rank, MM)
        return

    def eval_approx_error_complete(self,M=None):
        self.approx_error=TT.error_TT_data_complete(self.approx_data,self.data, M)
        return

    def eval_approx_error(self,M=None,Norm="L2"):
        self.approx_error=TT.error_TT_data(self.approx_data,self.data, M, Norm)
        return


def QTT_SVD(A,q,tol=1e-6):
    """ Take any ndarray A and approxinates it with quantic q QTT SVD, returns approximation"""
    qA=QuanticsTensor(A)
    qA.reshape_to_q(q)

    qA.applyTTSVD(eps=tol)
    return qA

def approx_with_QTT_SVD(A,q,tol=1e-6):
    """ Take any ndarray A and approxinates it with quantic q QTT SVD, returns approximation and it's metrics"""
    qA=QTT_SVD(A,q,tol=1e-6)
    qA.eval_approx_error_complete()
    return qA

def run_test(d=3,N=64,q=2):
    from analysis.plot import benchmark_norm_plotter

    A=np.random.rand(N,N,N)
    qA=approx_with_QTT_SVD(A,q)
    print(qA)
    benchmark_norm_plotter(qA.approx_error)
    return

if __name__=="__main__":
    run_test()
    # print(np.linalg.norm(TT.to_full()-TT.to_full([1,2,3,1])))
    # print("Error eval:",error_TT_data(TT,TT.to_full()))

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
from time import time
import pickle
from core.tensor_algebra import norm
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
        self.has_been_extended=False
        self.Approx_data=None # can be either a numpy array (init) or pointing to a TensorTrain object
        self.Approx_error=None
        self.size=data.size
        self.ndim=len(data.shape)
        self.rank=None

    def __str__(self):
        str="QuanticsTT\n"
        str+="ndim={}\n".format(self.ndim)
        str+="original_shape={}\n".format(self.original_shape)
        str+="q={}\n".format(self.q)
        str+="qshape={}\n".format(self.qshape)
        # str+="data=\n{}\n".format(self.data)
        str+="-------------------------------------------\n"
        str+="Approx_data rank=\n{}\n".format(self.Approx_data.rank)
        return str

    def reshape_to_q(self,q):
        #testing size and q fit
        D=math.log( self.size, q )
        self.q=q
        if not D.is_integer() :
            print("Warning data size ({}) is not a power of q ({}), filling with zeros till {}".format(self.size,q, q**(math.ceil(D))))
            self.reshape_any_to_q(type="flatten")
        else:
            self.qshape=[q] * int(D)
            print("reshaping to {}".format(self.qshape))
            self.data=np.reshape(self.data,self.qshape)

        return

    def reshape_any_to_q(self,type="flatten"):
        if type == "flatten":
            q=self.q
            D=math.log( self.size, q )
            self.qshape=[q] * int(math.ceil(D))
            newData=np.zeros(q**int(math.ceil(D)))
            newData[:self.size]=np.ravel(self.data)
            self.data=np.reshape(newData,self.qshape)

        elif type=="per_dim":
            raise SystemError("itertools not available on my machine")
            # Construct the base "q" shape
            # self.qshape = [ self.q**(int(math.log(s-0.5,self.q))+1)  for s in self.original_shape ]
            # # Resize the array and fill the extended dimensions
            # # with data on the -1 hyper-faces
            # Anew = np.zeros(self.qshape)
            # Anew[ tuple([ slice(0,gs,None) for gs in self.original_shape ]) ] = self.data
            # for dcube in range(self.ndim):
            #     cubes = itertools.combinations(range(self.ndim), dcube+1)
            #     for cube in cubes:
            #         idxs_out = []
            #         idxs_in = []
            #         for i, gs in enumerate(self.original_shape):
            #             if i in cube:
            #                 idxs_out.append( slice(gs,None,None) )
            #                 idxs_in.extend( [-1,np.newaxis] )
            #             else:
            #                 idxs_out.append( slice(0,gs,None) )
            #                 idxs_in.append( slice(0,gs,None) )
            #         idxs_out = tuple(idxs_out)
            #         idxs_in = tuple(idxs_in)
            #         Anew[ idxs_out ] = self.data[ idxs_in ]
            #     self.data = Anew

            # Set the folded_shape (list of list) for each dimension
            # self.folded_shape = [ [self.q] * \
            #                       int(round(math.log(self.data.shape[i],self.q)))
            #                       for i in range(len(self.original_shape)) ]
            #
            # # Folding matrix
            # new_shape = [self.base] * int(round(math.log( self.data.size, self.q )))
            # self.data = self.data.reshape(new_shape)
            print(new_shape)
            self.has_been_extended=True
        return

    def reshape_manual(self,qshape):
        raise NotImplementedError("not programmed yet")
        return

    def applyTTSVD(self,eps=1e-3,rank=-1,MM=None, cutoff=0.9):
        # I have observed some comvergence issue, trying with EVD first, then PRIMME
        try:
            self.Approx_data=TT_SVD(self.data, eps, rank, MM,QTTcutoff=cutoff)
        except:
            print("in applyTTSVD, EVD didn't converge, tyring PRIMME")
            self.Approx_data=TT_SVD(self.data, eps, rank, MM, solver='PRIMME',cutoff=cutoff)
        self.rank=self.Approx_data.rank
        return

    def eval_approx_error_complete(self,M=None,sampling="quadratic"):
        self.Approx_error=TT.error_TT_data_complete(self.Approx_data,self.data, M, sampling=sampling)
        return

    def eval_approx_error(self,M=None,Norm="L2"):
        self.Approx_error=TT.error_TT_data(self.Approx_data,self.data, M, Norm)
        return
    
    def to_full(self,trunc_rank=[]):
        return self.Approx_data.to_full(trunc_rank)
    
    def save_to_file(self,path):
        #copy might not fit in memory
        try:
            QTT_light=self
            QTT_light.data=None
        except:
            self.data=None
            QTT_light=self
        file = open(path,'wb')
        pickle.dump(QTT_light,file)
        file.close()
        return
    
    def save(self,path):
        "simple alias of save_to_file"
        self.save_to_file(path)
        return

def QTT_SVD(A,q,tol=1e-6,maxramk=-1,cutoff=1,verbose=1):
    """ Take any ndarray A and approxinates it with quantic q QTT SVD, returns approximation"""
    start=time()
    qA=QuanticsTensor(A)
    qA.reshape_to_q(q)

    qA.applyTTSVD(eps=tol,cutoff=cutoff)
    if verbose > 0:
        print("QTT_SVD walltime : {:.2f}s")
    return qA

def approx_QTT_SVD_epilon_based(A,q,tolmin=1e-2,tolmax=1e-16,cutoff=1):
    """ Take any ndarray A and approxinates it with quantic q QTT SVD, returns approximation stats
    for a list of epsilons"""
    qA=QuanticsTensor(A)
    qA.reshape_to_q(q)
    A_volume=np.product(A.shape)
    eps_list=[1e-1,1e-2,1e-3,1e-4,1e-5, 1e-6,1e-7,1e-8,1e-9, 
              1e-10, 1e-11,1e-12,1e-14,1e-16]
    eps_list=[item for item in eps_list if (item>=tolmax and item <tolmin)]
    norm_T={"L1":norm(A,type="L1"),
            "L2":norm(A,type="L2"),
            "Linf":norm(A,type="Linf")}
    error={"L1":[],"L2":[],"Linf":[]}
    comp_rate=[]
    for eps in eps_list:
        print("Computing QTT for eps={}".format(eps))
        start=time()
        qA.applyTTSVD(eps=eps,cutoff=cutoff)
        print("Approximation walltime={:.2f}".format(time()-start))
        T_approx=qA.to_full()
        E=qA.data-T_approx
        error["L1"].append(norm(E,type="L1")/norm_T["L1"])
        error["L2"].append(norm(E,type="L2")/norm_T["L2"])
        error["Linf"].append(norm(E,type="Linf")/norm_T["Linf"])
        comp_rate.append(qA.Approx_data.mem_eval()/A_volume)
        if comp_rate[-1]>1:
            # no point continuing to compute if compression rate above 100
            break 
    return error, np.asarray(comp_rate)

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

"""
Created on 17/07/2018

@author: lucas

A bunch of numerical tests for the manuscript.
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:09:32 2018

@author: Diego Britez
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import time
import os

import utils.tensor_creator as tensor_creator
import core.tensor_algebra as ta
import core.MassMatrices as mm
from core.PGD import PGD
from core.tucker_decomp import HOPOD, SHOPOD, THOSVD, STHOSVD
from core.TT_SVD import TT_SVD
from core.RPOD import rpod, RecursiveTensor, plot_rpod_approx,rpod_error_data
from core.Canonical import CanonicalTensor, canonical_error_data
from core.Tucker import TuckerTensor, tucker_error_data
from core.TensorTrain import TensorTrain, error_TT_data
from core.QuanticsTT import QTT_SVD,QuanticsTensor
from core.hierarchical_decomp import HierarchicalTensor, HT_build_error_data

from analysis.plot import benchmark_plotter, several_d_plotter, benchmark_norm_plotter
from utils.tensor_creator import testg,testf
import utils.IO as IO

def numerics_for_thesis(test_list):
    # General comparison separable function SVD everywhere
    if "general_3D" in test_list:
        general_test_3D(["SVD","trapezes"])

    if "num_dim_test_short" in test_list:
        num_dim_test_short()

    if "num_dim_test_long" in test_list:
        num_dim_test_long()

    if "Vega" in test_list:
        Vega_test([1])

    if "grid_imbalance" in test_list:
        grid_imbalance_test([1,2])

    return


def Vega_test(cases):
    if 1 in cases:
        print("\n ===================================\
              \n Vega function test\n Case 1 \n ==============================")
        n=16
        d=5
        err_data={}
        # decomp_methods=["SHO_POD","SHO_SVD","TT_POD","TT_SVD"]
        # decomp_methods=["RPOD",'RSVD',"SHO_POD","SHO_SVD","TT_SVD","TT_POD"]
        decomp_methods=["HT","SHO_SVD",'TT_SVD',"QTT_SVD"]
        solver=["SVD" for i in range(len(decomp_methods))]
        path='../output/'
        shape=[n for x in range(d)]
        plot_name=path+'vega_all_methods_n={}.pdf'.format(n)
        plot_title="Vega function decomposition, n={}".format(n)
        err_data=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                            test_function="Vega", plot=True,output='../output/num_dim_test',
                            Frob_norm=True,  plot_name=plot_name,
                            tol=1e-6, plot_title=plot_title)
    if 2 in cases:
        print("\n ===================================\
              \n Vega function test\n Case 2 \n ==============================")
        n=40
        d=5
        err_data={}
        decomp_methods=["RPOD",'RSVD',"SHO_POD","SHO_SVD","TT_SVD","TT_POD"]
        solver=["trapezes" for i in range(len(decomp_methods))]
        path='../output/vega_func/'
        shape=[n for x in range(d)]
        plot_name=path+'vega_all_methods_case2.pdf'.format(d)
        plot_title="Vega function decomposition, n={}".format(n)
        err_data=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                            test_function="Vega", plot=False,output='../output/num_dim_test/',
                            Frob_norm=True,  plot_name=plot_name,
                            tol=1e-12, plot_title=plot_title)

def grid_imbalance_test(case):
    if 1 in case:
        print("\n ===================================\
              \n Grid imbalance test function test\n ==============================")

        shape=[1000,20,15,12]
        # shape=[20 for i in range(4)]
        d=4
        f=2
        err_data={}
        decomp_methods=['RPOD',"SHO_POD","TT_POD"]
        solver=["SVD" for i in range(len(decomp_methods))]
        path='../output/grid_imbalance/'
        plot_name=path+'grid_imbalance_1.pdf'
        plot_title="f_{} function decomposition, shape={}".format(f,shape)
        err_data=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                            test_function=2, plot=True,output='../output/num_dim_test/',
                            Frob_norm=True,  plot_name=plot_name,
                            tol=1e-12, plot_title=plot_title)

    if 2 in case:
        shape=[100,20,20,20,20]
        d=5
        err_data={}
        decomp_methods=['RSVD',"SHO_SVD","TT_SVD"]
        solver=["SVD" for i in range(len(decomp_methods))]
        path='../output/grid_imbalance/'
        plot_name=path+'grid_imbalance_2.pdf'
        plot_title="Vega function decomposition, shape={}".format(shape)
        err_data=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                            test_function=2, plot=True,output='../output/num_dim_test/',
                            Frob_norm=True,  plot_name=plot_name,
                            tol=1e-12, plot_title=plot_title)

def num_dim_test_short():
        print("\n ===================================\
              \nTest number of dimension, fixed n per dim\n")
        n=20
        err_data={}
        decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
        solver=["SVD","SVD","SVD","SVD","SVD"]
        path='../output/num_dim_test_short/'

        for d in range(2,5):
            print("===================\nd={}\n".format(d))
            shape=[n for x in range(d)]
            plot_name=path+'func_2_d_{}.pdf'.format(d)
            plot_title="f_2 decomposition, d={}, n={}".format(d,n)
            err_data[d]=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                                test_function=2, plot=False,output='../output/num_dim_test/',
                                Frob_norm=True,  plot_name=plot_name,
                                tol=1e-16, plot_title=plot_title)
        IO.save(err_data,path+"saved_decomp_data.dat")
        several_d_plotter(err_data, show=True,plot_name=path+"full_view.pdf")

def num_dim_test_long():
    print("\n ===================================\
          \nTest number of dimension, fixed n per dim\n")
    err_data={}
    decomp_methods=["RSVD","SHO_SVD","TT_SVD"]
    solver=["SVD","SVD","SVD","SVD"]
    path='../output/num_dim_test_long/'

    n=32
    for d in range(3,6):
        print("===================\nd={}\n".format(d))
        shape=[n for x in range(d)]
        plot_name=path+'func_2_d_{}.pdf'.format(d)
        plot_title="f_2 decomposition, d={}, n={}".format(d,n)
        err_data[d]=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                            test_function=2, plot=False,output='../output/num_dim_test/',
                            Frob_norm=True,  plot_name=plot_name,
                            tol=1e-16, plot_title=plot_title)
    IO.save(err_data,path+"saved_decomp_data.dat")
    several_d_plotter(err_data, show=True,plot_name=path+"full_view.pdf")


def general_test_3D(cases):
    print("\n ===================================\
    \nTest number of dimension, fixed n per dim\n")
    path ="../output/general_3D/"
    decomp_methods=["RPOD","HO_POD","SHO_POD","TT_POD","PGD"]

    if "SVD" in cases:
        solver=["SVD","SVD","SVD","SVD","SVD"]
        for f in [1,2,3]:
            print("\n F{} \n".format(f))
            plot_name=path+'approx_benchmark_function_{}_Frob.pdf'.format(f)
            plot_title="f_{} decomposition, d=3, n={}, Frob norm".format(f,32)
            multi_var_decomp_analysis(decomp_methods, solver ,shape=[32,32,32],
                                test_function=f, plot=False,output='../output',Frob_norm=False,
                                plot_name=plot_name,tol=1e-12)
    if "trapezes" in cases:
        solver=["trapezes" for i in range(5)]
        for f in [1,2,3]:
            print("\n F{} \n".format(f))
            plot_name=path+'approx_benchmark_function_{}_L2.pdf'.format(f)
            plot_title="f_{} decomposition, d=3, n={}, L2 norm".format(f,32)
            multi_var_decomp_analysis(decomp_methods, solver ,shape=[32,32,32],
                                test_function=f, plot=False,output='../output',Frob_norm=False,
                                plot_name=plot_name,tol=1e-12)

def multi_var_decomp_analysis(list_reduction_method, integration_methods,
                              shape,test_function=1, plot=False,show=False,
                              output=None,Frob_norm=False,
                              plot_name='output/approx_benchmark',
                              tol=1e-5, plot_title=None):
    """
    This is a useful routine for running multivariate decomposition algorithms.
    """
    dim=len(shape)
    domain=[0,1]
    acepted_reduction_method=['PGD','THO_SVD','STHO_SVD','HO_POD', 'SHO_POD',
                              'RPOD','TT_POD','RSVD','TT_SVD']
    acepted_integration_methods=['trapezes','SVD']
    number_plot=0
    approx_data={}

    number_of_methods=len(list_reduction_method)

    if type(plot_name)!= str:
        raise ValueError('output_variable_name must be a string variable')
    if type(plot) != bool :
        raise TypeError("Error!! wrong plot is a boolean.")
    show_plot=plot
    if type(plot_name)==str:
        if plot_name!='':
            plot=True

    #########################END OF TESTS##########################################

    X,F=testg(test_function, shape, dim, domain)
    for ii in range(number_of_methods):
        reduction_method=list_reduction_method[ii]
        integration_method=integration_methods[ii]

        if integration_method=='SVD':
            M=mm.MassMatrices([mm.identity_mass_matrix(x.size) for x in X])
            Frob_norm=True
        elif integration_method=='trapezes':
            M=mm.mass_matrices_creator(X)
        else:
            raise NotImplementedError("integration_method '{}' \
                                      not implemented".format(integration_method))

        t=time.time()
        print(reduction_method)
        if reduction_method=='PGD':
            Result=PGD(M,F,epenri=np.sqrt(tol))
        elif reduction_method=='HO_POD':
            Result=HOPOD(F,M,tol=tol)
        elif reduction_method=='SHO_POD':
            Result=SHOPOD(F,M,tol=tol)
        elif reduction_method=='THO_SVD':
            Result=THOSVD(F)
        elif reduction_method in ["SHO_SVD","ST_HOSVD","STHO_SVD"]:
            Result=STHOSVD(F,epsilon=tol)
        elif reduction_method=='RPOD':
            Result=rpod(F, int_weights=M, POD_tol=1e-16,cutoff_tol=tol)
        elif reduction_method=='TT_POD':
            Result=TT_SVD(F, tol, MM=M)
        elif reduction_method=='RSVD':
            Result=rpod(F, POD_tol=1e-16,cutoff_tol=tol)
        elif reduction_method=='TT_SVD':
            Result=TT_SVD(F, tol)
        elif reduction_method=='QTT_SVD':
            Result=QTT_SVD(F,2,tol=tol)
        elif reduction_method=="HT":
            eps_list=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-12,1e-13,1e-14]
            eps_list=[item for item in eps_list if item>tol]
            Result=HT_build_error_data(F,eps_list,mode="homogenous",eps_tuck=tol,rmax=200)
            approx_data[reduction_method]=np.stack([Result[0]["L2"],Result[1]])

        else:
            raise AttributeError("reduction_method : '{}' is not a valid method".format(reduction_method))
        print("{} decompostion time: {:.3f} s".format(reduction_method,time.time()-t))
        print("{} Full rank reconstruction time: {:.2f} s".format(
            reduction_method,reconstruction_time(Result)))
        t=time.time()
        if Frob_norm:
            M=None
        if plot:
            if type(Result)==TuckerTensor:
                approx_data[reduction_method]=np.stack(tucker_error_data(Result,F,M))
            elif type(Result)==RecursiveTensor:
                approx_data[reduction_method]=np.stack(rpod_error_data(Result,F,M=M, max_tol=1e-8))
            elif type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F,tol=tol,M=M))
            elif type(Result)==TensorTrain:
                approx_data[reduction_method]=np.stack(error_TT_data(Result,F,M=M))
            elif type(Result)==QuanticsTensor:
                Result.eval_approx_error(Norm="L2")
                approx_data[reduction_method]=np.stack(Result.Approx_error)
            print("{} Error evaluation time: {:.2f} s".format(reduction_method,time.time()-t))
            benchmark_norm_plotter(approx_data[reduction_method], show=show,
                plot_name="output_{}_decomp_error_{}d_powderscale".format(reduction_method,len(shape)),
                plot_title="Singular Function {} approximation errors"
                )
        try:
            if output!='':
                np.savetxt(output+"/"+reduction_method+".csv",
                  np.transpose([approx_data[reduction_method][0],approx_data[reduction_method][1]]),
                  delimiter=',')
        except:
            pass
    if plot:
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,title=plot_title)
    return approx_data

def reconstruction_time(T_reduced):
    t=time.time()
    if type(T_reduced)==TuckerTensor:
        T_reduced.reconstruction()
    elif type(T_reduced)==RecursiveTensor:
        T_reduced.to_full()
    elif type(T_reduced)==CanonicalTensor:
        T_reduced.to_full_quick()
    elif type(T_reduced)==TensorTrain:
        T_reduced.to_full()
    t=time.time()-t
    return t

if __name__ == '__main__':
    avail_test=["general_3D","num_dim_test_short","num_dim_test_long","Vega",'grid_imbalance']
    # test_list=["general_3D","Vega"]
    test_list=avail_test[-2]
    numerics_for_thesis(test_list)

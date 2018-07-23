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

from analysis.plot import benchmark_plotter, several_d_plotter
from utils.tensor_creator import testg,testf
import utils.IO as IO

def numerics_for_thesis(test_list):
    # General comparison separable function SVD everywhere
    if "general_3D" in test_list:
        print("\n ===================================\
        \nTest number of dimension, fixed n per dim\n")
        path ="../output/general_3D/"
        decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
        solver=["SVD","SVD","SVD","SVD","SVD"]
        for f in [1,2,3]:
            print("\n F{} \n".format(f))
            plot_name=path+'approx_benchmark_function_{}_Frob.pdf'.format(f)
            plot_title="f_{} decomposition, d=3, n={}, Frob norm".format(f,32)
            multi_var_decomp_analysis(decomp_methods, solver ,shape=[32,32,32],
                                test_function=f, plot=False,output='../output',Frob_norm=False,
                                plot_name=plot_name,tol=1e-12)

        solver=["trapezes" for i in range(5)]
        for f in [1,2,3]:
            print("\n F{} \n".format(f))
            plot_name=path+'approx_benchmark_function_{}_L2.pdf'.format(f)
            plot_title="f_{} decomposition, d=3, n={}, L2 norm".format(f,32)
            multi_var_decomp_analysis(decomp_methods, solver ,shape=[32,32,32],
                                test_function=f, plot=False,output='../output',Frob_norm=False,
                                plot_name=plot_name,tol=1e-12)
    ###########################################################################
    if "num_dim_test_short" in test_list:
        print("\n ===================================\
              \nTest number of dimension, fixed n per dim\n")
        n=20
        err_data={}
        decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
        solver=["SVD","SVD","SVD","SVD","SVD"]
        path='../output/num_dim_test_short/'

        for d in range(2,6):
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


    if "num_dim_test_long" in test_list:
        print("\n ===================================\
              \nTest number of dimension, fixed n per dim\n")
        err_data={}
        decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD"]
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

    return

def multi_var_decomp_analysis(list_reduction_method, integration_methods,
                              shape,test_function=1, plot=False,
                              output=None,Frob_norm=False,
                              plot_name='output/approx_benchmark',
                              tol=1e-5, plot_title=None):
    """
    This is a useful routine for running multivariate decomposition algorithms.
    """
    dim=len(shape)
    domain=[0,1]
    acepted_reduction_method=['PGD','THO_SVD','STHO_SVD','HO_POD', 'SHO_POD','RPOD','TT_SVD']
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
        if reduction_method=='PGD':
            Result=PGD(M,F,epenri=np.sqrt(tol))
        elif reduction_method=='HO_POD':
            Result=HOPOD(F,M,tol=tol)
        elif reduction_method=='SHO_POD':
            Result=SHOPOD(F,M,tol=tol)
        elif reduction_method=='THO_SVD':
            Result=THOSVD(F)
        elif reduction_method=='STHO_SVD':
            Result=STHOSVD(F)
        elif reduction_method=='RPOD':
            Result=rpod(F, int_weights=M, POD_tol=1e-16,cutoff_tol=tol)
        elif reduction_method=='TT_SVD':
            Result=TT_SVD(F, tol, MM=M)
        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

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
        try:
            if output!='':
                np.savetxt(output+"/"+reduction_method+".csv",np.transpose([approx_data[reduction_method][0],approx_data[reduction_method][1]]), delimiter=',')
        except:
            pass

    if plot:
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,title=plot_title)
    return approx_data

if __name__ == '__main__':
    avail_test=["general_3D","num_dim_test_short","num_dim_test_long"]
    test_list=avail_test[1]
    numerics_for_thesis(test_list)

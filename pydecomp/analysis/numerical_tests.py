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

from analysis.plot import benchmark_plotter
from utils.tensor_creator import testg,testf

def numerics_for_thesis(test_list):
    # General comparison separable function SVD everywhere
    if "general_3D" in test_list:
        for f in [1,2,3]:
            decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
            solver=["SVD","SVD","SVD","SVD","SVD"]
            plot_name='../output/approx_benchmark_function_{}.pdf'.format(f)
            plot_title="f_{} decomposition, d=3, n={}".format(f,32)
            multi_var_decomp_analysis(decomp_methods, solver ,shape=[32,32,32],
                                test_function=f, plot=False,output='../output',Frob_norm=False,
                                plot_name=plot_name,tol=1e-12)

    ###########################################################################
    if "num_dim_test_short" in test_list:
        print("\n ===================================\
              \nTest number of dimension, fixed n per dim\n")
        err_data={}
        for d in range(2,6):
            print("===================\nd={}\n".format(d))
            n=20
            shape=[n for x in range(d)]
            decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
            solver=["SVD","SVD","SVD","SVD","SVD"]
            plot_name='../output/num_dim_test/func_2_withPGD_d_{}.pdf'.format(d)
            plot_title="f_2 decomposition, d={}, n={}".format(d,n)
            err_data[d]=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                                test_function=2, plot=False,output='../output/num_dim_test/',
                                Frob_norm=True,  plot_name=plot_name,
                                tol=1e-6, plot_title=plot_title)


    if "num_dim_test_long" in test_list:
        print("\n ===================================\
              \nTest number of dimension, fixed n per dim\n")
        err_data={}
        for d in range(2,7):
            print("===================\nd={}\n".format(d))
            n=20
            shape=[n for x in range(d)]
            decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD"]#,"PGD"]
            solver=["SVD","SVD","SVD","SVD"]#,"SVD"]
            plot_name='../output/num_dim_test/func_2_d_{}.pdf'.format(d)
            plot_title="f_2 decomposition, d={}, n={}".format(d,n)
            err_data[d]=multi_var_decomp_analysis(decomp_methods, solver ,shape=shape,
                                test_function=2, plot=False,output='../output/num_dim_test/',
                                Frob_norm=True,  plot_name=plot_name,
                                tol=1e-6, plot_title=plot_title)

    return

def multi_var_decomp_analysis(list_reduction_method, integration_method,
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

    if type(list_reduction_method) == str:
        list_reduction_method=[list_reduction_method]
    number_of_methods=len(list_reduction_method)

    if type(integration_method)==str:
         list_integration_method=[integration_method for i in range(number_of_methods)]

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
        integration_method=list_integration_method[ii]

        if integration_method=='SVD':
            M=mm.MassMatrices([mm.identity_mass_matrix(x.size) for x in X])
            Frob_norm=True
        elif integration_method=='trapezes':
            M=mm.mass_matrices_creator(X)

        t=time.time()
        if reduction_method=='PGD':
            Result=PGD(M,F,epenri=tol)
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
            Result=TT_SVD(F, tol)
        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

        if Frob_norm:
            M=None
        if plot:
            if type(Result)==TuckerTensor:
                approx_data[reduction_method]=np.stack(tucker_error_data(Result,F,M))
            elif type(Result)==RecursiveTensor:
                approx_data[reduction_method]=np.stack(rpod_error_data(Result,F,M=M, max_tol=tol))
            elif type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F,tol=tol))
            elif type(Result)==TensorTrain:
                approx_data[reduction_method]=np.stack(error_TT_data(Result,F))
        try:
            if output!='':
                np.savetxt(ouput+"/"+reduction_method+".csv",approx_data[reduction_method][0],
                           approx_data[reduction_method][1],delimiter=',')
        except:
            print("Failed to save data")
            pass

    if plot:
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,title=plot_title)
    return approx_data

if __name__ == '__main__':
    avail_test=["general_3D","num_dim_test_short","num_dim_test_long"]
    test_list=avail_test[:2]
    numerics_for_thesis(test_list)

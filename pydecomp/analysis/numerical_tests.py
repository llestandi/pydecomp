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

from plot import benchmark_plotter
from utils.tensor_creator import testg,testf

def multi_var_decomp_analysis(list_reduction_method, integration_method,
                              shape,test_function=1, plot=False,
                              output=None,
                              plot_name='output/approx_benchmark', tol=1e-5):
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
    elif type(integration_method)==list:
         list_integration_method=integration_method
         if len(integration_method)!=number_of_methods:
             error_number_of_integration_methods="""  The number of integration
             methods are not coherent with the number  of the list of reduction
             methods  """
             raise ValueError(error_number_of_integration_methods)
    else:
        error_integration_method="""  Variable integration method must be 'SVD'
        or 'trapezes' strig variable or  a list of these two options as many
        times as the number of integration  methods selected.  """
        raise ValueError(error_integration_method)


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
            X=[np.ones(x) for x in shape]
            M=[diags(x) for x in X]
        elif integration_method=='trapezes':
            M=mm.mass_matrices_creator(X)

        print(type(M))
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
            Result=TT_SVD(F, tol)
        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

        if plot:
            if type(Result)==TuckerTensor:
                approx_data[reduction_method]=np.stack(tucker_error_data(Result,F))
            elif type(Result)==RecursiveTensor:
                approx_data[reduction_method]=np.stack(rpod_error_data(Result,F))
            elif type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F))
            elif type(Result)==TensorTrain:
                approx_data[reduction_method]=np.stack(error_TT_data(Result,F))
        try:
            if output!='':
                np.savetxt(reduction_method+".csv",approx_data[reduction_method][0],
                           approx_data[reduction_method][1],delimiter=',')
        except:
            pass

    if plot:
        benchmark_plotter(approx_data, show_plot)
    return approx_data

if __name__ == '__main__':
    decomp_methods=["RPOD","SHO_POD","SHO_POD","TT_SVD"]
    # decomp_methods=["HO_POD"]
    solver=["trapezes","trapezes","trapezes","SVD"]
    multi_var_decomp_analysis(decomp_methods, solver ,shape=[16,16,16,16,16],
                            test_function=3, plot=True,output='../output',
                            plot_name='output/approx_benchmark.pdf',tol=1e-8)

# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:09:32 2018

@author: Diego Britez
"""
import numpy as np
from scipy.sparse import diags
import time
from analysis.plot import mode_1D_plot, rank_benchmark_plotter
from analysis.benchmark_multivariable import testf

import utils.tensor_creator as tensor_creator
import core.tensor_algebra as ta
import utils.MassMatrices as mm
from core.Canonical import CanonicalTensor, canonical_error_data
from core.cls_POD import cls_POD, init_POD_class_from_decomp, pod_error_data
from core.PGD import PGD
from core.POD import POD
from core.TSVD import TSVD

def benchmark_2D(list_reduction_method, shape,test_function=1, plot=False,
                plot_name='output/approx_benchmark', tol=1e-5):
    """
    This function allows to see how the differents functions in the python
    decomposition library work. Differents equations (1 to 3) are avaible
    in order to create synthetic data. \n

    **Parameters** \n

    reduction_method--> string type or list with string type elements.  \n
    Options expected: 'PGD','SVD','POD','SVD_by_EVD'.
    #-----------------------------------------------------------------------
    test_function--> Ingeger type. Values expected: 1 to 5. \n
    Option 1: 1/(1+X1^2+X2^2....+Xn^2)        \n
    Option 2: sin((X1^2+X2^2+...+Xn^2)^(0.5)) \n
    Option 3: X1xX2x...xXn                    \n
    Option 3: fill                            \n
    Option 3: fill                            \n
    #-----------------------------------------------------------------------
    shape--> List type with integers as elements. Each element will indicate
    the numbers of elements to divide de subspace domain. The number of
    elements must coincide with "dim" parameter value.
    Example= [32,50]  (2 Dimensions ) \n
    #-----------------------------------------------------------------------
    plot--> bool.
    Ables and unables the  error vs relative compression indice  graphic
    output.  \n
    #-----------------------------------------------------------------------
    """
    dim=len(shape)
    domain=[0,1]
    acepted_reduction_method=['PGD','SVD','POD','SVD_by_EVD']
    approx_data={}

    if type(list_reduction_method) == str:
        list_reduction_method=[list_reduction_method]
    number_of_methods=len(list_reduction_method)


    if type(plot) != bool :
        raise TypeError("Error!! wrong plot is a boolean.")
    show_plot=plot
    if type(plot_name)==str:
        if plot_name!='':
            plot=True
    else:
        raise ValueError('output_variable_name must be a string')

    X,F=testf(test_function, shape, dim, domain)

    #########################END OF TESTS##########################################
    for ii in range(number_of_methods):
        reduction_method=list_reduction_method[ii]
        if not (reduction_method in acepted_reduction_method):
            raise AttributeError("Wrong reduction method: "+reduction_method)

        if reduction_method in ['POD','PGD']:
            M=mm.mass_matrices_creator(X)

        t=time.time()
        if reduction_method=='PGD':
            Result=PGD(M,F,epenri=tol)
        elif reduction_method=='POD':
            Result=POD(F,M[0],M[1],tol=tol)
        elif reduction_method=='SVD':
            Result=TSVD(F,tol)
        elif reduction_method=='SVD_by_EVD':
            Result=TSVD(F,tol,solver='EVD')
        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

        if plot:
            if type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F,rank_based=True))
            else:
                approx=init_POD_class_from_decomp(Result[0],Result[1],Result[2])
                approx_data[reduction_method]=np.stack(pod_error_data(approx,F))
    if plot:
        rank_benchmark_plotter(approx_data, show_plot, plot_name)

    return

if __name__ == '__main__':
    decomp_methods=["POD","PGD","SVD","SVD_by_EVD"]
    benchmark_2D(decomp_methods ,shape=[32,3200], test_function=2, plot=True,
                plot_name='../output/2D_approx_benchmark.pdf',tol=1e-16)

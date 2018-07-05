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
        raise ValueError('output_variable_name must be a string variable')

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
            Result=PGD(M,F,epenri=np.sqrt(tol))
        elif reduction_method=='POD':
            Result=POD(F,M[0],M[1],tol=tol)
        elif reduction_method=='SVD':
            Result=TSVD(F,tol)
        elif reduction_method=='SVD_by_EVD':
            Result=TSVD(F,tol)
        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

        if plot:
            if type(Result)==cls_POD:
                approx=init_POD_class_from_decomp(Result)
                approx_data[reduction_method]=np.stack(pod_error_data(approx,F))
            elif type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F))
                # plot_error_canonical(Result,F, number_plot,label_line)
                # raise NotImplementedError("Canonical plot V2 is not implemented yet")

    if plot:
        benchmark_plotter(approx_data, show_plot)
    return

def rank_benchmark_plotter(approx_data, show=True, plot_name="plots/benchmark.pdf",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles=['r+-','b*-','ko-','gh:','mh--']
    fig=plt.figure()
    xmax=1
    ylim=[0.1,0.1]
    k=0
    plt.yscale('log')
    plt.xlabel("rank")
    plt.ylabel('Relative Error')
    plt.grid()

    for label, err in approx_data.items():
        ranks=np.arange(err.size)
        xmax=max(xmax,rank[-1])
        ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
        ax=fig.add_subplot(111)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ylim)
        plt.plot(ranks, err , styles[k], label=label)

        k+=1
    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name)
    plt.close()

def testf(test_function, shape, dim, domain ):
    """
    This function propose several models of function to be chosen to build a
    multivariable tensor (synthetic data) , the output
    depends only on the function selected (1,2..) and the number of dimention
    to work with.\n
    **Parameters**\n
    test_function: integer type, describes the format of the function selected:
    \n
    Formule 1 \n
    :math:`1\(1+X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2})`
    \n
    Formule 2 \n
    :math:`\sin(\sqrt {X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2}})`
    \n
    Formule 3 \
    :math:`X_{1}*X_{2}*...*X_{n}`
    \n
    **shape**: array or list type number of elements to be taken
    in each subspace. \n
    **dim**:Integer type. Number of dimentions.
    """
    test_function_possibilities=[1,2,3]
    if test_function not in test_function_possibilities:
        note="""
        Only 3 multivariable test functions are defined, please introduce
        introduce a valid value.
        """
        raise ValueError(note)
    if test_function==1:
        equation='(1'
        for i in range(dim):
            if i<dim-1:
               aux='+V['+str(i)+']**2'
               equation=equation+aux
            else:
               aux='+V['+str(i)+']**2)'
               equation=equation+aux
        equation='1/'+equation
    elif test_function==2:
        equation='np.sin(np.sqrt('
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']**2+'
                equation=equation+aux
            else:
                aux='V['+str(i)+']**2)'
                equation=equation+aux
        equation=equation+')'
    elif test_function==3:
        equation=''
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']*'
                equation=equation+aux
            else:
                aux='V['+str(i)+']'
                equation=equation+aux


    Function=tensor_creator.TensorCreator()
    #Creating the variables required for TensorCreator from the data
    lowerlimit=np.ones(dim)
    lowerlimit=lowerlimit*domain[0]
    upperlimit=np.ones(dim)
    upperlimit=upperlimit*domain[1]
    Function.lower_limit=lowerlimit
    Function.upper_limit=upperlimit
    Function.tshape=shape
    X,F= Function._Generator2(equation)

    return X,F


if __name__ == '__main__':
    decomp_methods=["POD","PGD","SVD","SVD_by_EVD"]
    benchmark_2D(decomp_methods ,shape=[32,32], test_function=2, plot=True,
                plot_name='output/2D_approx_benchmark.pdf',tol=1e-16)

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
from core.QuanticsTT import QuanticsTensor, QTT_SVD
from core.hierarchical_decomp import HierarchicalTensor, HT_build_error_data

from analysis.plot import benchmark_plotter

def benchmark_multivariable(list_reduction_method, integration_method,
                              shape,test_function=1, plot=False,
                              output_decomp=[],
                              plot_name='output/approx_benchmark', tol=1e-5,
                              which_norm="L2",**kwargs):
    """
    This function allows to see how the differents functions in the python
    decomposition library work. Differents equations (1 to 3) are avaible
    in order to create synthetic data. \n
    Between the outputs option are offered=  plot of the evolution of error vs
    relative  compression indice, binary file of the compressed object and
    the compressed object itself. \n
    **Parameters** \n

    reduction_method--> string type or list with string type elements.  \n
    Options expected: 'PGD','THO_SVD','STHO_SVD','HO_POD',TT_SVD or 'SHO_POD'.
    Example= 'PGD'     \n
    If an evaluation to compare two differents methods is wanted, this variable
    should be introduced as a list, with the methods to evaluate as the
    elements.
    If neither of this options are selected, and error message will appear. \n
    Example= ['PGD', 'STHO_SVD'] \n
    #-----------------------------------------------------------------------
    integration_method--> string type or list with string type elements if
    multiple methods are evaluated.
    Options expected: 'trapezes' or 'SVD'.
    If only one string is provided as an input and multiple methods are being
    used, the integration_method introduced is going to be used in each method
    if possible.
    If the integration method is not compatible with the reduction_method
    applied with selected,  an automatic change of method is going to be
     a message informing this action. \n
    #-----------------------------------------------------------------------
    dim-->Integer type. Number if dimentions or variables in the problem.\n

    #-----------------------------------------------------------------------
    test_function--> Ingeger type. Values expected: 1,2 or 3. \n
    Option 1: 1/(1+X1^2+X2^2....+Xn^2)        \n
    Option 2: sin((X1^2+X2^2+...+Xn^2)^(0.5)) \n
    Option 3: X1xX2x...xXn                    \n
    #-----------------------------------------------------------------------
    shape--> List type with integers as elements. Each element will indicate
    the numbers of elements to divide de subspace domain. The number of
    elements must coincide with "dim" parameter value.
    Example= [32,50,80]  (for a 3d case) \n
    #-----------------------------------------------------------------------
    plot--> String type. Expected input= 'yes' or 'no'.
    Ables and unables the  error vs relative compression indice  graphic
    output.  \n
    #-----------------------------------------------------------------------

    output_variable_file--> String type. Expected input= 'yes' or 'no'.
    Ables and unables the creation of a binary file with the compressed object
    created. \n
    #-----------------------------------------------------------------------
    output_variable_name--> string type or list of strings type elements.
    Name of the file/s created to contain the binary information of the
    object/s created.
    If there is only one string as an input, and multiple files created
    (in the case that more than one method is being evaluated), the multiples
    output_variable_name are going to be created from the single input string.
    #-----------------------------------------------------------------------
    Example:
    import benchmarck_mutivariable as bm \n

    bm.benchmarck_multivariable(['PGD','STHO_SVD','HO_POD'],'trapezes',3
    [40,25,30],test_function=2,plot='yes',output_variable_name='example')
    Binary file saved as'example(0)'\n

                For STHOSVD reduction method, integration method must be 'SVD',
                this change is automatically applied. \n

    Binary file saved as'example(1)' \n
    Binary file saved as'example(2)' \n



    """
    dim=len(shape)
    domain=[0,1]
    acepted_reduction_method=['PGD','THO_SVD','STHO_SVD','HO_POD', 'QTT_SVD', 'SHO_POD','RPOD','TT_SVD','HT','HT_2']
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
             error_number_of_integration_methods="""
             The number of integration methods are not coherent with the number
             of the list of reduction methods
             """
             raise ValueError(error_number_of_integration_methods)
    else:
        error_integration_method="""
        Variable integration method must be 'SVD' or 'trapezes' strig variable or
        a list of these two options as many times as the number of integration
        methods selected.
        """
        raise ValueError(error_integration_method)


    acepted_shape_type=[list]
    if type(shape) not in acepted_shape_type:
        error_shape_type="""
        shape input must be either a list or np.ndarray with integer as
        its elements.
        """
        raise ValueError(error_shape_type)


    if number_of_methods>1:
        if type(output_decomp)==str:
            list_output_variable_name=[]
            for i in range(number_of_methods):
                aux=output_decomp+'('+str(i)+')'
                list_output_variable_name.append(aux)
        elif type(output_decomp)==list:
            if len(output_decomp)!=number_of_methods:
                error_variable_name="""
                The number of output_variable_name is not coherent with the
                number of integration methods selected.
                """
                raise TypeError(error_variable_name)


    if type(plot) != bool :
        raise TypeError("Error!! wrong plot is a boolean.")
    show_plot=plot
    if type(plot_name)==str:
        if plot_name!='':
            plot=True

    X,F=testf(test_function, shape, dim, domain)

    for ii in range(number_of_methods):
        reduction_method=list_reduction_method[ii]
        integration_method=list_integration_method[ii]

        if reduction_method not in acepted_reduction_method:
            error_reduction_method="""{0} is incorrect.
            The acepted reduction methods are: {1}.
            Please verify and choose one value method correcly. """.format(
                    reduction_method,acepted_reduction_method)
            raise ValueError(error_reduction_method)

        if integration_method not in acepted_integration_methods:
            error_integration_method=" The only integration methods acepted are trapezes and SVD."
            raise ValueError(error_integration_method)


        if type(plot_name)!= str:
            raise ValueError('output_variable_name must be a string variable')


        if reduction_method=="THO_SVD":
            if integration_method != "SVD":
                note_print1="""
                For THO_SVD reduction method, integration method must be 'SVD',
                this change is automatically applied.
                """
                print(note_print1)
                integration_method='SVD'

        if  reduction_method=='TT_SVD':
             if integration_method != "SVD":
                note_print1="""
                For TT_SVD reduction method, integration method must be 'SVD',
                this change is automatically applied.
                """
                print(note_print1)
                integration_method='SVD'

        if reduction_method=="STHO_SVD":
            if integration_method != "SVD":
                note_print2="""
                For STHOSVD reduction method, integration method must be 'SVD',
                this change is automatically applied
                """
                print(note_print2)
                integration_method="SVD"
#########################END OF TESTS##########################################



        if integration_method=='SVD':
            X=[np.ones(x) for x in shape]
            M=[diags(x) for x in X]
        elif integration_method=='trapezes':
            M=mm.mass_matrices_creator(X)

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
            Result=STHOSVD(F,tol=tol)
        elif reduction_method=='RPOD':
            Result=rpod(F, int_weights=M, POD_tol=1e-16,cutoff_tol=tol)
        elif reduction_method=='TT_SVD':
            Result=TT_SVD(F, tol)
        elif reduction_method=='QTT_SVD':
            try:
                Result=QTT_SVD(np.copy(F),2,tol=tol)
            except:
                print("QTT_SVD has been removed since it did not converge. \n It is very sensitive to conditionning and may not converge")
                continue
        elif reduction_method=='HT_2':
            eps_list=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-12,1e-13,1e-14]
            eps_list=[item for item in eps_list if item>tol]
            Result=HT_build_error_data(F,eps_list,eps_tuck=tol,rmax=200)
            # by construction we need to evaluate errors before hand
            approx_data[reduction_method]=np.stack([Result[0][which_norm],Result[1]])
        if reduction_method=="HT":
            eps_list=[1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-12,1e-13,1e-14]
            eps_list=[item for item in eps_list if item>tol]
            Result=HT_build_error_data(F,eps_list,mode="homogenous",eps_tuck=tol,rmax=200)
            approx_data[reduction_method]=np.stack([Result[0][which_norm],Result[1]])

        print("{} decompostion time: {:.2f} s".format(reduction_method,time.time()-t))

        if plot:
            if type(Result)==TuckerTensor:
                approx_data[reduction_method]=np.stack(tucker_error_data(Result,F,Norm=which_norm,sampling="exponential_fine"))
            elif type(Result)==RecursiveTensor:
                approx_data[reduction_method]=np.stack(rpod_error_data(Result,F,Norm=which_norm))
            elif type(Result)==CanonicalTensor:
                approx_data[reduction_method]=np.stack(canonical_error_data(Result,F,Norm=which_norm))
            elif type(Result)==TensorTrain:
                approx_data[reduction_method]=np.stack(error_TT_data(Result,F,Norm=which_norm,sampling="exponential_fine"))
            elif type(Result)==QuanticsTensor:
                Result.eval_approx_error(Norm=which_norm)
                approx_data[reduction_method]=np.stack(Result.Approx_error)
                print(approx_data[reduction_method])

                # plot_error_canonical(Result,F, number_plot,label_line)
                # raise NotImplementedError("Canonical plot V2 is not implemented yet")
        try:
            if output_decomp!='':
                ta.save(Result,output_decomp)
        except:
            pass

    if plot:
        benchmark_plotter(approx_data, show_plot, plot_name,**kwargs)
    return




#------------------------------------------------------------------------------
def testg(test_function, shape, dim, domain ):
    # @Diego FIXME
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
    Formule 2 \
    :math:`1\(1+X_{1}^{2}X_{2}^{2}...X_{n}^{2})`
    \n
    Formule 3 \n
    :math:`\sin(\sqrt {X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2}})`
    \n
    **shape**: array or list type number of elements to be taken
    in each subspace. \n
    **dim**:Integer type. Number of dimentions.
    """
    Function=tensor_creator.TensorCreator()
    #Creating the variables required for TensorCreator from the data
    Function.lowerlimit=np.ones(dim)*domain[0]
    Function.upperlimit=np.ones(dim)*domain[1]
    Function.tshape=shape
    X,F= Function._Generator(test_function)
    print(X,F)
    return X,F

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
        equation='1./(np.sqrt(1.1 -'
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']*'
                equation+=aux
            else:
                aux='V['+str(i)+']'
                equation=equation+aux
        equation+="))"
        print(equation)
    elif test_function==4:
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
    decomp_methods=["PGD"]
    solver=["trapezes"]
    # decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD","PGD"]
    # solver=["trapezes","trapezes","trapezes","SVD",'trapezes']
    benchmark_multivariable(decomp_methods, solver ,shape=[32,32,32],
                            test_function=1, plot=True,output_decomp='',
                            plot_name='output/approx_benchmark.pdf',tol=1e-16)
                            # plot_name='')

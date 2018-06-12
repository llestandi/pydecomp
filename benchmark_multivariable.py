# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:09:32 2018

@author: Diego Britez
"""
import tensor_creator
import numpy as np
import high_order_decomposition_method_functions as hf
from scipy.sparse import diags
from pgd_main import PGD
from HOPOD import HOPOD
from SHOPOD import SHOPOD
from THOSVD import THOSVD
from STHOSVD import STHOSVD
from TT_SVD import TT_SVD
from plot_error_canonical import plot_error_canonical
from plot_error_tucker import plot_error_tucker
from plot_error_tt import plot_error_tt

def benchmark_multivariable(list_reduction_method, integration_method,
                              dim, shape,test_function=1, plot="no",
                              output_variable_file='yes',
                              output_variable_name='variable'):



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
    domain=[0,1]
    #Number plot is just a variable that will define ploting characteristics
    #such as color, linestyle, marker
    number_plot=0

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


    if number_of_methods>1:
        if type(output_variable_name)==str:
            list_output_variable_name=[]
            for i in range(number_of_methods):
                aux=output_variable_name+'('+str(i)+')'
                list_output_variable_name.append(aux)
        elif type(output_variable_name)==list:
            if len(output_variable_name)!=number_of_methods:
                error_variable_name="""
                The number of output_variable_name is not coherent with the
                number of integration methods selected.
                """
                raise TypeError(error_variable_name)
    elif number_of_methods==1:
        list_output_variable_name=[output_variable_name]


    for ii in range(number_of_methods):
        reduction_method=list_reduction_method[ii]
        integration_method=list_integration_method[ii]
        output_variable_name=list_output_variable_name[ii]



        acepted_shape_type=[list]
        if type(shape) not in acepted_shape_type:
            error_shape_type="""
            shape input must be either a list or np.ndarray with integer as
            its elements.
            """
            raise ValueError(error_shape_type)

        acepted_output_variable_file_option=['yes', 'no']
        if output_variable_file not in acepted_output_variable_file_option:
            error_output_variable_file="""
            Fatal error in output_variable_file!!! \n
            output_variable_file acepts either 'yes' or 'no' as an input option
            """
            raise ValueError(error_output_variable_file)

        if type(output_variable_name)!=str:
            error_output_variable_name="""
            Fatal error in output_variable_name!!! \n
            String type variable is spected.
            """
            raise ValueError(error_output_variable_name)

        acepted_reduction_method=['PGD','THO_SVD','STHO_SVD','HO_POD','SHO_POD','RPOD']
        if reduction_method not in acepted_reduction_method:
            error_reduction_method="""
            The acepted reduction methods are: 'PGD',THO_SVD','STHO_SVD',
            'HO_POD','SHO_POD'. \n
            Please verify and choose one value method correcly.
            """
            raise ValueError(error_reduction_method)

        acepted_integration_methods=['trapezes','SVD']

        if integration_method not in acepted_integration_methods:
            error_integration_method="""
            The only integration methods acepted are trapezes and SVD.
            """
            raise ValueError(error_integration_method)


        if len(shape)!=dim:
             error="""
             Number of elements of shape and "dim" value must be equals
             """
             raise ValueError(error)


        plot_options=['yes', 'no']
        if plot not in plot_options:
            plot_error="""
            Error!! wrong plot argument. plot='yes' or plot='no' expected.
            """
            raise TypeError(plot_error)


        output_variable_file_options=['yes', 'no']
        if output_variable_file not in output_variable_file_options:
            output_variable_file_error="""
            Error!! wrong plot argument. output_variable_file='yes' or
            output_variable_file='no' expected.
            """
            raise ValueError(output_variable_file_error)

        if type(output_variable_name)!= str:
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
                
        if reduction_method=='PGD':
            if integration_method=='SVD':
                note_error_integration_method="""
                PGD reduction method does not work with SVD mass matrices,
                trapezes method will be automatically applied
                """
                print(note_error_integration_method)
                integration_method='trapezes'



        if reduction_method=="STHO_SVD":
            if integration_method != "SVD":
                note_print2="""
                For STHOSVD reduction method, integration method must be 'SVD',
                this change is automatically applied
                """
                print(note_print2)
                integration_method="SVD"

        X,F=testf(test_function, shape, dim, domain)

        if integration_method=='SVD':
            X=[np.ones(x) for x in shape]

        if integration_method=='trapezes':
            M=hf.mass_matrices_creator(X)
        if integration_method=='SVD':
            M=[diags(x) for x in X]



        if reduction_method=='PGD':
            Result=PGD(M,F)


            if plot=='yes':
                number_plot=number_plot+1
                label_line="PGD"
                plot_error_canonical(Result,F, number_plot,label_line)

        if reduction_method=='HO_POD':
            Result=HOPOD(F,M)
            if plot=='yes':
                number_plot=number_plot+1
                label_line='HO_POD'
                plot_error_tucker(Result,F,number_plot,label_line,
                                  output_variable_name='variable')

        if reduction_method=='SHO_POD':
            Result=SHOPOD(F,M)
            if plot=='yes':
                number_plot=number_plot+1
                label_line=reduction_method
                plot_error_tucker(Result,F,number_plot,label_line,
                                  output_variable_name='variable')

        if reduction_method=='THO_SVD':
            Result=THOSVD(F)
            if plot=='yes':
                number_plot=number_plot+1
                label_line=reduction_method
                plot_error_tucker(Result,F,number_plot,label_line,
                                  output_variable_name='variable')

        if reduction_method=='STHO_SVD':
            Result=STHOSVD(F)
            if plot=='yes':
                number_plot=number_plot+1
                label_line=reduction_method
                plot_error_tucker(Result,F,number_plot,label_line,
                                  output_variable_name='variable')
        
        if reduction_method=='TT_SVD':
            Result=TT_SVD(F, eps=1e-10, rank=100)
            if plot=='yes':
                number_plot=number_plot+1
                label_line=reduction_method
                plot_error_tt(Result, F, number_plot=number_plot, 
                               label_line=label_line,
                               output_variable_name='variable') 
                            
                
            
        if  output_variable_file=='yes':
            hf.save(Result,output_variable_name)







#------------------------------------------------------------------------------
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

    if test_function==2:
        equation='np.sin(np.sqrt('
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']**2+'
                equation=equation+aux
            else:
                aux='V['+str(i)+']**2)'
                equation=equation+aux
        equation=equation+')'

    if test_function==3:
        equation=''
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']*'
                equation=equation+aux
            else:
                aux='V['+str(i)+']'
                equation=equation+aux

    if test_function not in test_function_possibilities:
        note="""
        Only 3 multivariable test functions are defined, please introduce
        introduce a valid value.
        """
        raise ValueError(note)


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

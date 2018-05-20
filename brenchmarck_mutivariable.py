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
from plot_error_canonical import plot_error_canonical
from plot_error_tucker import plot_error_tucker

def brenchmark_multivariable(reduction_method,integration_method,
                              dim, shape,test_function=1, plot="no", 
                              output_variable_file='yes',
                              output_variable_name='variable'):
                             
                            
                             
    domain=[0,1]
    
    acepted_shape_type=[numpy.ndarray, list]
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
    
    acepted_reduction_method=['PGD','THOSVD','STHOSVD','HOPOD','SHOPOD']
    if reduction_method not in acepted_reduction_method:
        error_reduction_method="""
        The acepted reduction methods are: 'PGD',THOSVD','STHOSVD','HOPOD',
        'SHOPOD'. \n 
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
    
    
    if reduction_method=="THOSVD":
        if integration_method != "SVD":
            note_print1="""
            For THOSVD reduction method, integration method must be 'SVD',
            this change is automatically applied.
            """
    if reduction_method=='PGD':
        if integration_method=='SVD':
            note_error_integration_method="""
            PGD reduction method does not work with SVD mass matrices, 
            trapezes method will be automatically applied
            """
            print(note_error_integration_method)
            integration_method='trapezes'
    
            print(note_print1)
            integration_method="SVD"
    
    if reduction_method=="STHOSVD":
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
            plot_error_canonical(Result,F)
    
    if reduction_method=='HOPOD':
        Result=HOPOD(F,M)
        if plot=='yes':
            plot_error_tucker(Result,F)
    
    if reduction_method=='SHOPOD':
        Result=SHOPOD(F,M)
        if plot=='yes':
            plot_error_tucker(Result,F)
    
    if reduction_method=='THOSVD':
        Result=HOPOD(F,M)
        if plot=='yes':
            plot_error_tucker(Result,F)
    
    if reduction_method=='STHOSVD':
        Result=SHOPOD(F,M)
        if plot=='yes':
            plot_error_tucker(Result,F)
        
        
    
        
    
    return Result
                                   
    
    
    
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
    
    
    
    
    
    
    
    
    
                                 
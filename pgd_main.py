# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018

@author: Diego Britez
"""
 
import numpy as np
from fixepoint import fixpoint
from scipy.linalg import norm
import high_order_decomposition_method_functions as hf
from Canonical import CanonicalForme
import timeit
import integrationgrid




def PGD(X,F, epenri=1e-10, maxfix=15):
    #start = timeit.default_timer()
    """
    This function use the PGD method for a multivariate problem decomposition,
    returning an Canonical class objet.
    \n
    **Parameters**:\n
    **X**: This parameter is the list with all the vectors (as 1 dimention
     array). It is the cartesian grid that describes the distribution of F
     in the space.\n
    **F**: ndarray type, it's the tensor that its decomposition is wanted.\n
    **epenri**: Stop criteria value for the enrichment loop, default value=1e-10.
    This value is obtained by the divition  of the first mode with the 
    mode obtained in the last fixed-point iteration of the last variable
    (wich carries all the information of the module of the function),  more
    detailed information can be find in  Lucas Lestandi's thesis.
    
    **maxfix**:is the maximal number of iteration made inside of the fix point 
    function before declare that the method has no convergence.
    Default value=15 \n
    
    **Introduction to the PGD method**\n
  
    Be :math:`F` the tensor that is going to be decomposed by the PGD method.
    And be 
    :math:`u^{p}`
    the solution, so we can express:\n
    :math:`u^{p}=\sum_{m=1}^{p}\prod_{i=1}^{D}X_{i}^{p}(x_{i})`.\n
    The weak  formulation of the problem can be expressed as:\n
    :math:`\int_{\Omega}u^{*}(u-F)=0`, is the test function, \n
    for all :math:`u^{*} \in H^{1}(\Omega)`. If we take;\n
    :math:`u^{*}=\prod_{i=1}^{s-1} X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})`,\n
    the weak formulation becomes:\n
    :math:`\int_{\Omega}[\prod_{i=1}^{s-1}X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})(u^{p-1}+\prod_{i=1}^{s}X_{i}^{k+1}
    (x_{i})\prod_{i=s+1}^{D}X_{i}^{k}(x_{i})-F)]=0` \n
                                    
    """
    
    
    tshape=F.shape
    dim=len(tshape)  
    
          
    #Start value of epn that allows get in to the enrichment loop for the first
    #iteration
    
    epn=1                             
     
                             
    #The variable z is created to count the number of iterations in the 
    #fix point loop which don't arrive to the convergence after maximal 
    #times of iterations declared as a stoping criteria.
    
    z=0                               
    
                                
    #Xgrid is a variable that is used for integration functions
    Xgrid=integrationgrid.IntegrationGrid(X,dim,tshape)   
    Xgrid=Xgrid.IntegrationGridCreation()
    
                                                            
    #The Verification variable is going to be used to evaluate the orthogonality
    #Verification=1 ----> Orthogonality verified
    #Verification=0 ----> Non Orthogonality found
                                     
    Verification=1
    C=CanonicalForme(tshape,dim)
    
    C.solution_initialization()
                            

    while   (epn>=epenri):
        C.set_rank()
            
        
        
        R,z=fixpoint(X,C._tshape,C._U,F,z ,C._rank,maxfix)        
        C.add_enrich(R)                                                                       
        
        #Unmark  next line if the orthogonality verification is desired
        #Verification=hf.orth(dim,Xgrid,R,C)
        
        
        if C.get_rank()==1:
            REF=R[dim-1]
           
        epn=norm(R[dim-1])/norm(REF)
        
    #Unmark commentary if Orthogonality Verification is desired
    """
    if (Verification==1):   
                print('Orthogonality between modes was verified') 
                print('----------------------------------------\n')
    print('--------------------------------------------------------------\n')      
    print("Iteration's enrichment loops=",C.get_rank()) 
    print("Iteration's enrichment loops in fixed-point loop with no convergence:",z)       
    print('Epsilon=',epn)
    """
    
    #Eliminating the first (zeros) row created to initiate the algorithm
    C._U=[x[1:,::] for x in C._U]
    #C.writeU() 
    #stop=timeit.default_timer()
    return C


      
        
  
 
 



     




    
    




      
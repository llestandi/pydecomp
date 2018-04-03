# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018

@author: Diego Britez
"""






 
import numpy as np

from fixepoint import fixpoint
from scipy.linalg import norm
import matplotlib.pyplot as plt
from matrixf import matfunc
from new_iteration_result import new_iteration_result
from orthogonality_verification import orthogonality_verification
from Orthogonal import Orthogonal
from IPython import get_ipython
from Canonical import CanonicalForme
from evolution_error import evolution_error
import CartesianGrid
import timeit
import csv
import integrationgrid




def PGD(X,F, epenri=1e-10, maxfix=15):
    #start = timeit.default_timer()
    """
    This function use the PGD method for the reduction of models.
    The X variable is the list with all the vectors who describe the space
    of study and data distribution of the Tensor F. This two variables can
    be obtained from equation if wanted with the functino tensor_creator.
    """
    
    """
    epenri: Stop criteria value for the enrichment loop, default value=1e-10.
    This value is obtained by the divition  of the first mode with the 
    mode obtained in the last fixed-point iteration of the last variable
    (wich carries all the information of the module of the function),  more
    detailed information can be find in the Manuscript of Lucas Lestandi
    
    maxfix=is the maximal number of iteration made in fixed point method before
    declare that the method has no convergence. Default value=30
    
    
                                  
    """
    tshape=F.shape
    dim=len(tshape)  
    
    """       
    Start value of epn that allows get in to the enrichment loop for the first
    iteration
    """
    epn=1                             
     
    """                          
    The variable z is created to count the number of iterations in the fixed-point
    loop which don't arrive to the convergence after maximal times of iterations
    #declared as a stoping criteria.
    """
    z=0                               
    
    """                              
    Xgrid is a variable that is used for integration functions
    """
    Xgrid=integrationgrid.IntegrationGrid(X,dim,tshape)   
    Xgrid=Xgrid.IntegrationGridCreation()
    
    """                                                         
    The Verification variable is going to be used to evaluate the orthogonality
    Verification=1 ----> Orthogonality verified
    Verification=0 ----> Non Orthogonality found
    """                                 
    Verification=1                   

               
    

    C=CanonicalForme(tshape,dim)
    #evaluation=evolution_error()
    C._solution_initialization()
    #Resultat=np.zeros(tshape)
                        

    while   (epn>=epenri):
        C._set_rank()
            
            
       
        if (C._rank==1 ):
          print('Number of iterations in fixed-point routine for each enrichment')
          print('--------------------------------------------------------------\n')  
        
        R,z=fixpoint(X,C._tshape,C._U,F,z ,C._rank,maxfix)
        
        #Sn=S/norm(S)
        """
        #-------------------------------------------------------------
        for i in range(dim):
            
            Orth=orthogonality_verification(Xgrid[i],R[i],C._U[i])
            for j in range(C._get_rank()):
                if (Orth[j]>1e-10):
                    print('--------------------------------------------------------------\n')
                    print('Orthogonality is not verified at mode:', C._get_rank())
                    print('Variable of failure= U(',i,')')
                    print('Value of the scalar product of failure=', Orth[j])
                    Verification=0
                
        
        #--------------------------------------------------------------        
        """
        C._add_enrich(R)                                                                       
        #Resultat=new_iteration_result(R,tshape,Resultat)
        
        if C._get_rank()==1:
            REF=R[dim-1]
           
        epn=norm(R[dim-1])/norm(REF)
        
        
        #ERROR=norm(F-Resultat)/norm(F)
        """
        This routine serve to evaluate the evolution of the value of 
        error for each enrichment loop, if the evaluation is not desired,
        leave this section as a comentary
        """
        
        #evaluation._new_loop_evaluation(C._rank,ERROR)
        
       
        if (((norm(F-Resultat)/norm(F)))<(1e-12)):
            print('--------------------------------------------------------------\n') 
             
            print('Max. Relative error criteria was verified')
            print('Relative error=',(norm(F-Resultat)/norm(F)),'< 1e-10' )
            break
        

    if (Verification==1):   
                print('Orthogonality between modes was verified') 
                print('----------------------------------------\n')
    print('--------------------------------------------------------------\n')      
    print("Iteration's enrichment loops=",C._get_rank()) 
    print("Iteration's enrichment loops in fixed-point loop with no convergence:",z)       
    print('Epsilon=',epn)



    ##--------------------------------------------------------------------------
    #For the simplicity in the code we name p to the rank number (number of modes)
    p=C._get_rank()

    #_final_result method is called to remove the first mode of zeros created
    #at the begining to initiate the enrichment process.
    C._final_result()


     


    """
    Creation of the files with each mode of the solution in independent files
    calling a mode in CanonicalForme
    """

    C._writeU() 

    """
    With the folowing command  all variables are going to be erased and the 
    solution modes could have have acces by the files that were created
    """ 
    return C
    #evaluation._plot_error()
    #stop = timeit.default_timer()
    #print (stop - start)
    #get_ipython().magic('reset -sf') 
    
 



     




    
    




      
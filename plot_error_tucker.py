# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:15:06 2018

@author: Diego Britez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
import Tucker
import sys

def plot_error_tucker(A,F):
    """
    This function will calculate the error in each mode from a decomposed
    form  of an object in the tucker class and plot it in funtion to the
    number of modes.\n
    
    **Parameters**
   
    A--> Data to be evaluated, it should be an Tucker class object. \n 
    
    
    F--> mxk original matrix or data Matrix, numpy array type of data.\n
    
    **Returns**
    Plot of Relative Error vs Average Rank
    
    """
    #We are going to calculate one average value of ranks
    dim=len(A.u)
    average_rank=1
    data_compression=[]
    F_volume=fvolume(F)
    #In n_u we storage the number of elements in each subspace
    n_u=F.shape
   
    for i in range(dim):
        average_rank=average_rank*np.shape(A.u[i][1])[0]
    
    average_rank=(average_rank)**(1/dim)
    average_rank=np.round_(average_rank)
    average_rank=average_rank.astype(int)
    #print(average_rank)
    
    #Creating a list with the maximal rank for each subspace
    r=[]
    #Creating a list where the errors obtained in each iteration will be 
    #storaged
    error=[]
    #in storage_weight variable the object class weight is going to be storaged
    #to evaluate the data reduction. 
    storage_weight=[]
   
    for i in range(dim):
        aux=np.shape(A.u[i][1])[0]
        if aux<average_rank:
            r.append(aux)
        else:
            r.append(average_rank)
    
    
    #Calculating the error in each enrichment
    #Creating the projections for each enrichment
    for i in range(average_rank):
        shape_core=np.zeros(dim)
        
        projections=[]
        for j in range(dim):            
            if r[j]>=i:
                aux=A.u[j][::,:i+1]
                projections.append(aux)
               
            else:
                aux=A.u[j][::,:r[j+1]]
                projections.append(aux)
                
            if r[j]>i:
                shape_core[j]=int(i+1)
            else:
                shape_core[j]=int(r[j]) 
                
        #Calculatin the volume of the compressed data
        actual_core_volume=np.prod(shape_core)
        
        projections_volume=[]
        for j in range(dim):
            aux=shape_core[j]*n_u[j]
            projections_volume.append(aux)
        actual_projections_volume=sum(projections_volume)
        actual_volume=actual_projections_volume+actual_core_volume
        storage_weight.append(actual_volume)
        
    
        #Here the we search the shape that the core will be sliced 
        for j in range(dim):
            if j<(dim-1):
                Cutting_core_shape=":"+str(int(shape_core[j]))+","
            else:
                Cutting_core_shape=":"+str(int(shape_core[j]))
            if j==0:
                Cutting_core_shape_complet=Cutting_core_shape
            if j>0:
                Cutting_core_shape_complet=Cutting_core_shape_complet+Cutting_core_shape
        Cutting_core_shape_complet="A.core["+Cutting_core_shape_complet+"]"
        
        AUX_CORE=eval(Cutting_core_shape_complet)
        
        #Creating a tucker object from the truncated data
        
        AUX_TUCKER=Tucker.Tucker(AUX_CORE,projections)
        
        #Finding the solution for the actual truncated Tucker object
        
        Fparcial=AUX_TUCKER.reconstruction()
        
        #Fparcial is a FullFormat class object, the tensor is stored in _Var
        #element of this class
        
        Fparcial=Fparcial._Var
        
        actual_error=norm(F-Fparcial)/norm(F)
        error.append(actual_error)
        modes=np.arange(1,average_rank+1)
        
    #Evaluating data compression    
    data_compression=[x/F_volume*100 for x in storage_weight]    
        
        
    """  
    #plotting the results error vs rank
    
    plt.ylim(1e-10, 0.15)
    plt.xlim(0,average_rank+2)
    plt.plot(modes, error ,color='k', marker='^',linestyle='-',
             label='Relative error F1_3D \n with THOSVD method')
        
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()
    """
    
    #ploting the results error vs relative weight
    plt.ylim(1e-16, 0.15)
    plt.xlim(0,150)
    plt.plot(data_compression, error ,color='b', marker='^',linestyle='-',
             label='Relative error F1_3D \n with STHOSVD method')
        
    plt.yscale('log')
    plt.xlabel("Data volume in %")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()
    
    
def actual_weight(dim, projections, core):
   projections_weight=0  
   for j in range(dim):
          aux=sys.getsizeof(projections[j])
          projections_weight=projections_weight+aux
   weight_core=sys.getsizeof(core)
   weight=projections_weight+weight_core
   return weight
        
#Calculation of Tensor volume
def fvolume(F):
    Fshape=F.shape
    volume=1
    for i in range(len(Fshape)):
        volume=volume*Fshape[i]
    return volume        
        
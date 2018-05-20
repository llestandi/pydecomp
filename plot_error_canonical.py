# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:00:38 2018

@author: Diego Britez
"""
import Canonical
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
    
def plot_error_canonical(A,F):    
    """
    This function will calculate the error in each mode from a decomposed
    form  of an object in the tucker class and plot it in funtion to the
    number of modes.\n
    
    **Parameters**
   
    A--> Data to be evaluated, it should be a Canonical class object.\n
    
    F--> mxk original matrix or data Matrix, numpy array type of data.\n
    
    **Returns**
    Plot of Relative Error vs  Rank
    
    """

    dim=A._dim
    rank_max=A._rank
    #Creating a list where the errors obtained in each iteration will be 
    #storaged
    error=[]
    tshape=A._tshape
    AUX_CANONICAL=Canonical.CanonicalForme(tshape,dim)
    modes=np.arange(1,rank_max+1)
    data_compression=[]
    F_volume=fvolume(F)
    for i in range(rank_max):
        
        for j in range(dim):
            if i==0: 
                
                Uaux=A._U[j][i]
                Uaux=Uaux.reshape([1,tshape[j]])
                
                AUX_CANONICAL._U.append(Uaux)
            else: 
                Uaux=A._U[j][i]
                Uaux=Uaux.reshape([1,tshape[j]])
                AUX_CANONICAL._U[j]=np.append(AUX_CANONICAL._U[j],Uaux,axis=0)
        
        #Reconstructing the partials solutions
        Fpartial=AUX_CANONICAL.reconstruction()
        #Calculating the actual reduction 
        
        data_compression.append(data_volume(AUX_CANONICAL,dim))
        
        
        actual_error=norm(Fpartial-F)/norm(F)
        error.append(actual_error)
        
    data_compression=[x/F_volume*100 for x in data_compression]
        
    """
    #plotting the results error vs rank
    
    plt.ylim(error[rank_max-1], 0.15)
    plt.xlim(0,rank_max+2)
    plt.plot(modes, error ,color='k', marker='^',linestyle='-',
             label='Relative error F1_3D \n with PGD method')
        
    plt.yscale('log')
    plt.xlabel("Rank")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()    
    """
    
    #plotting the results error vs data_compression
    
    plt.ylim(1e-10, 0.15)
    plt.xlim(0,150)
    plt.plot(data_compression, error ,color='k', marker='^',linestyle='-',
             label='Relative error F1_3D \n with PGD method')
        
    plt.yscale('log')
    plt.xlabel("Data volume in %")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()
    
    




#Calculation of the data_volume of a Canonical object         
def data_volume(A,dim):
    aux=0
    for i in range(dim):
        auxi=A._U[i].shape
        auxi=auxi=auxi[0]*auxi[1]
        aux=aux+auxi
       
    return aux

#Calculation of Tensor volume
def fvolume(F):
    Fshape=F.shape
    volume=1
    for i in range(len(Fshape)):
        volume=volume*Fshape[i]
    return volume
            
            
        
    
    
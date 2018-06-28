# -*- coding: utf-8 -*-
"""
Created on Thu May 17 11:00:38 2018

@author: Diego Britez
"""
import Canonical
import numpy as np
from scipy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
    
def plot_error_canonical(A,F, number_plot=1, label_line='PGD',
                         output_variable_name='variable'):    
    """
    This function will reconstruct and calculate the error for each mode from 
    a decomposed form  of an object in the canonical class and plot it in 
    function to the compression rate (%).\n
    
    **Parameters** \n
   
    A--> Data to be evaluated, it should be a Canonical class object.\n
    F--> mxk original matrix or data Matrix, numpy array type of data.\n
    number_plot--> integer type (from 1 to 5) that will define the figure 
    properties (line style, color, markers). \n
    label_line--> string type, label to characterise the curve 
    
    **Returns**
    Plot of Relative Error vs  Compression rate 
    
    """
    
    dim=A._dim
    rank_max=A._rank
    if type(number_plot)!=int:
        number_plot=4
    if number_plot<1:
        number_plot=2
    if number_plot>5:
        number_plot=1
    if number_plot==1:
        color_code='r'
        marker_code='+'
        linestyle_code= '-'        
    if number_plot==2:
        color_code='b'
        marker_code='*'
        linestyle_code='--'
    if number_plot==3:
        color_code='k'
        marker_code='o'
        linestyle_code='-.'
    if number_plot==4:
        color_code='g'
        marker_code='h'
        linestyle_code=':'
    if number_plot==5:
        color_code='m'
        marker_code='h'
        linestyle_code='--'
    
    
        
    #Creating a list where the errors obtained in each iteration will be 
    #storaged
    error=[]
    tshape=A._tshape
    AUX_CANONICAL=Canonical.CanonicalFormat(tshape,dim)
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
        
    #Calculating the relative data reduction for each mode    
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
    if plt.fignum_exists(1):
        if data_compression[-1]>plt.axis()[1]:
            plt.xlim(0,(data_compression[-1]+5))
    
                  
    else:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(0,(data_compression[-1]+5))
        ax.set_ylim([1e-16,0.15])
        
   
    plt.plot(data_compression, error ,color=color_code, marker=marker_code,
             linestyle=linestyle_code,
             label=label_line)
        
    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()
    plt.legend()
    
    #saving plot as pdf
    pp = PdfPages(output_variable_name+'.pdf')
    plt.savefig(pp, format='pdf')
    pp.savefig()
    pp.close()

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
            
            
        
    
    
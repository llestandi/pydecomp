# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 12:49:43 2018

@author: Diego Britez
"""

import numpy as np
from scipy.linalg import norm
from functools import reduce
from operator import mul
from TensorTrain import TensorTrain
import matplotlib.pyplot as plt

def plot_error_tt(GG,F,number_plot=1,label_line="",
                      output_variable_name='variable'):
    error=[]
    dim=GG.dim
    Fshape=F.shape
    sizeF=reduce(mul,Fshape)
    compressedlist=[]
    ranks=GG.ranks
    rank_max=max(ranks)
    
          
    for i in range(rank_max):
        G=GG.G[:]
        dim=len(G)
        if i+1<= ranks[0]:
            G[0]=G[0][:,:,:(i+1)]
          
        else:
            G[0]=G[0][:,:,:(ranks[0])]
                 
        for j in range(1,dim-1):
          
            if (i+1<=ranks[j-1] and (i+1)<=ranks[j]):
                
                G[j]=G[j][:(i+1),:,:(i+1)]
                         
            elif ((i+1)<=ranks[j-1] and (i+1)>ranks[j]):
                
                G[j]=G[j][:(i+1),:,:(ranks[j])]
                 
            elif (i+1)>(ranks[j-1]) and (i+1)<=ranks[j]:
                G[j]=G[j][:(ranks[(j-1)]),:,:(i+1)]
                   
            elif ((i+1)>ranks[j-1] and (i+1)>ranks[j]):
                G[j]=G[j][:(ranks[(j-1)]),:,:(ranks[j])]
           
        if (i+1)<=ranks[-2]:
            G[-1]=G[-1][:(i+1),:,:]
                  
        else:
            G[-1]=G[-1][:(ranks[-2]),:,:]
        #Creating a new TensorTrain element for each enrichment loop
        New_TensorTrain_Object=TensorTrain(G)          
        NewF=New_TensorTrain_Object.reconstruction()
       
        #Calcul of actual error
        errori=norm(NewF-F)/norm(F)
        error.append(errori)
        
        #Calcul Compresion
        compression=np.zeros(dim)
        for u in range(dim):
            compression[u]= reduce(mul,G[u].shape)
        compression=sum(compression)/sizeF*100
        compressedlist.append(compression)
        
        #Ploting setup
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
            
    if plt.fignum_exists(1):
            if compressedlist[-1]>plt.axis()[1]:           
                plt.xlim((0,compressedlist[-1]+10))    
                
       
                      
    else:
           fig=plt.figure()
           ax=fig.add_subplot(111)
           ax.set_xlim(0,compressedlist[-1]+10)
           ax.set_ylim([1e-16,0.2])    
            
            
    #ploting the results error vs relative weight
        
    plt.plot(compressedlist, error ,color=color_code, marker=marker_code,
                 linestyle=linestyle_code,
                 label=label_line)
            
    plt.yscale('log')
    plt.xlabel("Compression rate")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.show()
    plt.ion()
    plt.legend()        
        
    """
    #saving to pdf
    pp = PdfPages(output_variable_name+'.pdf')
    plt.savefig(pp, format='pdf')
    pp.savefig()
    pp.close()
    """    
    
        

   
        
        
        
        
        
    
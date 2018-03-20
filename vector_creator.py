# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:36:53 2018

@author: Diego Britez
"""
import numpy as np
"""
This function will create n dimention space vectors where the number of dimentions
is given by the variable dim, and the number of divitions of each space is 
contained in a vector calles div_tshape. The number of elements of div_tshape
must coincide with the number of dim, else  the function will call to an error,
similar error is going to be given if lower_limit or upper_limit dont coincide
with dim value.  
"""
def vector_creator(lower_limit,upper_limit,div_tshape,dim):
    #div_tshape=np.array([5,5])
    if (np.size(div_tshape) != dim):
        print('The number of elements of div_tshape and dim must be equals')
    if (np.size(lower_limit) != dim):  
        print('The number of elements of lower_limit and dim must be equals')
    if (np.size(upper_limit) != dim):
        print('The number of elements of upper_limit and dim must be equals')
    X=np.array([np.zeros(div_tshape[0])])
    
    for i in range (1,dim):
        X=np.append(X,[np.zeros(div_tshape[i])],axis=0)


    for i in range (dim):
        X[i]=np.linspace(lower_limit[i],upper_limit[i],div_tshape[i])
        
    return X
        
if __name__=="__main__":
    lower_limit = np.array([0,0])
    upper_limmit = np.array([1,1])
    div_tshape=np.array([5,5])
    dim=2     
    X=vector_creator(lower_limit,upper_limmit,div_tshape,dim)
    print(X)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:50:53 2018

@author: Diego Britez
"""
import numpy as np

def integration_1dtrap(f,x):
    """This function will integrate the 'f' vector discretised in the x domain
    f= nd array represetation of the function 
    x= nd array representation of the grid of the domain
    """
    nx=x.size
    """
    if (nx!=f.size):
        
        print('Integration-1dtrap fonction and domaine dont have same dimentions','\n',
              'dimention of the fonction', f.size,'\n', 'dimention of the discretitation grid',
              x.size)
     """    
    
    w=np.zeros(nx)
    w[1:-1]=(x[2:]-x[0:-2])/2
    w[0]=(x[1]-x[0])/2
    w[-1]=(x[-1]-x[-2])/2
    Int =np.dot(f,w)
    
    
    return Int

def test(x):
    return x**1

#------------------------------------------------------------------------------

if __name__=="__main__":
    nx=51
    x=np.linspace(0,1,nx)     #Sert a créer une grille entre 0 et 1 avec nx éléments
    print(x)
    f=test(x)
    print(f)
    F=integration_1dtrap(f,x)
    print(F)
    print(F-1./2)
  
    

  

    
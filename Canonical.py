# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:43:55 2018

@author: Diego Britez
"""
import numpy as np
import csv

class CanonicalForme:
#This module is the code where  the atributs and methodes of the 
#class CanonicalForme are defined. CanonicalForme is the class where all  the  
#information relative to the variables of te PGD method are storaged, 
#such as the modes, the number of  dimentions, the
#rank (number of enrichment or modes).    
    def __init__(self,nx,ny):        
        self._d=2                                         #Number of dimentions
        self._rank=0                                      #Tensor rank
        self._tshape=np.array([nx,ny])
        self._U=[[np.zeros(self._tshape[0])],[np.zeros(self._tshape[1])]]
        self._Sum=np.zeros(shape=(ny,nx))
#------------------------------------------------------------------------
#Method to get   _rank value       
    def _get_rank(self):
        return self._rank
#Method to set a new _rank value
    def _set_rank(self):
        self._rank=self._rank+1
#------------------------------------------------------------------------
#Method to get _tshape value
    def _get_tshape(self):
        return self._tshape
#Method to set a new _tshape value
    def _set_tshape(self,tshape):
            self._tshape=tshape
#------------------------------------------------------------------------           

#_U is the tensor  such as _U=[RR,SS] where RR is the tensor with the 
#solutions modes vectors (aranged in rows) in X (called also R)space 
#and SS is the tensor with te solutions modes vectors in Y (called also S).
#_get_U is the methode to call and get _U
#_set_U is the methode to call and get _U
#_add_enrich is the method to add a new mode in RR and SS in each enrichmet 
#loop   
  
    def _get_U(self):
        return self._U
    def _set_U(self,newU):
        self._U=newU
        
    def _add_enrich(self,R,S):
        self._U[0]=np.append(self._U[0],R,axis=0)
        self._U[1]=np.append(self._U[1],S,axis=0)
#------------------------------------------------------------------------  
#The variable _Sum is the variable where the partial result at each enrichment
#loop is calculated and storaged. We use this variable to compare with the 
#function F as a stop criteria. This criteria can be aboided but ther are 
#going to be other enrichment loops before the principal criteria is reached, 
#it has to be studied wich is less computational consumer.        
    def _get_Sum(self):
        return self._Sum 
    def _set_Sum(self,Sumn):
        self._Sum=self._Sum+Sumn   

#--------------------------------------------------------------------------
#The methode writeUR serve to write the modes R into a file .csv (coma 
#separated values) 
    def _writeUR(self):
        with open('UR.csv','w',newline='') as fp:
            a=csv.writer(fp, delimiter=',')
            a.writerows(self._U[0]) 
            
#The methode writeUS serve to write the modes of S into a file .csv (coma
#separated values)
    def _writeUS(self):
        with open('US.csv','w',newline='') as fp:
            a=csv.writer(fp, delimiter=',')
            a.writerows(self._U[1])         
            
#----------------------------------------------------------------------------

#The method readUR serve to read the file UR.csv 
     
    def _readUR(self):
        return np.loadtxt("UR.csv",delimiter=",",dtype=float)
    
#----------------------------------------------------------------------------
        
#The method readUS serve to read the file UR.csv 
     
    def _readUS(self):
        return np.loadtxt("US.csv",delimiter=",",dtype=float)
    
#----------------------------------------------------------------------------
        
#Redefining the substractions for objects with Canonical forme   
    
    def __sub__(C,object2):
        
        if (isinstance(object2,CanonicalForme)==False):
            print('New object has not the correct canonical forme')
           
        if (C._d!=object2._d):
            print('Fatal error!, Objects dimentions are not equals!')
            
        if (C._tshape[0]!=object2._tshape[0]):
            print('Fatal error!, shapes of spaces are not compatibles!')
        if (C._tshape[1]!=object2._tshape[1]):
            print('Fatal error!, shapes of spaces are not compatibles!')
        
        
        New_Value=CanonicalForme(C._tshape[0],C._tshape[1]) 
        New_Value._rank=C._rank+object2._rank
        New_Value._d=C._d
        
        object2._U[0]=np.multiply(-1,object2._U[0])
        object2._U[1]=np.multiply(-1,object2._U[1])
        
        New_Value._U[0]=np.append(C._U[0],object2._U[0],axis=0)
        New_Value._U[1]=np.append(C._U[1],object2._U[1],axis=0)
        
        
        return New_Value     
    
class TensorDescriptor(CanonicalForme):
    pass

    
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:43:55 2018

@author: Diego Britez
"""
import numpy as np
from tensor_descriptor_class import TensorDescriptor
import csv
#------------------------------------------------------------------------
        
        
#This module is the code where  the atributs and methodes of the 
#class CanonicalForme are defined. CanonicalForme is the class where all  the  
#information relative to the variables of te PGD method are storaged, 
#such as the modes, the number of  dimentions, the
#rank (number of enrichment or modes)

class CanonicalForme(TensorDescriptor):
    
    def __init__(self,_tshape,dim):                                                                  
        TensorDescriptor.__init__(self,_tshape,dim)
        self._rank=0                                      
        self._U=[]
#------------------------------------------------------------------------
    def _solution_initialization(self):
        for i in range(self._dim):
            self._U.append(np.zeros(self._tshape[i]))
            self._U[i]=np.array([self._U[i]])
#------------------------------------------------------------------------
#This method is defined to delete the zeros rows that were used to initiate
#the list of solutions _U
    
    def _final_result(self):
        for i in range(self._dim):
            self._U[i]=np.delete(self._U[i],0,axis=0)
            
        
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
        
    def _add_enrich(self,R):
        for i in range(self._dim):
            """
            print('i',i)
            print('U',i,self._U[i])
            """
            
           
            self._U[i]=np.append(self._U[i],np.array([R[i]]),axis=0)
           # self._U[1]=np.append(self._U[1],S,axis=0)

#--------------------------------------------------------------------------
#The methode writeU serve to write the solution modes  into a file .csv (coma 
#separated values) 
    def _writeU(self):
        for i in range (self._dim):
            aux="U("+str(i)+").csv"
            aux2='readfile.read("'+aux+'")'
            print('--------------------------------------------------------------\n')
            print('The file', aux , 'has been created,to read it, type:\n',
                    aux2)
            
            with open(aux,'w',newline='') as fp:
                a=csv.writer(fp, delimiter=',')
                a.writerows(self._U[i]) 
            
            with open('tshape.csv','w', newline='') as file:
                b=csv.writer(file, delimiter=',')
                b.writerow(self._tshape)
            
    
#----------------------------------------------------------------------------
        
#Redefining the substractions for objects with Canonical forme   
    
    def __sub__(C,object2):
        
        if (isinstance(object2,CanonicalForme)==False):
            print('New object has not the correct canonical forme')
           
        if (C._d!=object2._d):
            print('Fatal error!, Objects dimentions are not equals!')
        
        for i in range(C._d):    
            if (C._tshape[i]!=object2._tshape[i]):
                print('Fatal error!, shapes of spaces are not compatibles!')
        #if (C._tshape[1]!=object2._tshape[1]):
        #    print('Fatal error!, shapes of spaces are not compatibles!')
        
        
        New_Value=CanonicalForme(C._tshape,C._dim) 
        New_Value._rank=C._rank+object2._rank
        New_Value._d=C._d
        
        for i in range(C._dim-1):
            object2._U[i]=np.multiply(-1,object2._U[i])
     
        for i in range(C._dim-1):
            New_Value._U[i]=np.append(C._U[i],object2._U[i],axis=0)
        
        
        
        return New_Value     
    




    
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:43:55 2018

@author: Diego Britez
"""
import numpy as np
from tensor_descriptor_class import TensorDescriptor
import csv
import pickle
import full_format_class
#------------------------------------------------------------------------
        

class CanonicalForme(TensorDescriptor):
    """  
          
    **Canonical Type Format**
    
    
    In this format, any tensor 
    :math:`\chi\in V = \otimes_{\mu=1}^{d}V_{\mu}`
    a tensor space, is written as the finite sum of rank-1 tensors.
    :math:`\chi \in C_{r} (\mathbb{R})`
    is said to be represented in the 
    canonical format and it reads, \n
    :math:`\chi=\sum_{i=1}^{r}{\otimes}_{\mu=1}^{d}x_{u,i}`
    \n
    **Attributes**
    
        **_U** : list type, in this list will be stored all the modes of the 
        decomposed matrix as an array type each mode (
        :math:`x_{u,i}`).\n
        **_tshape**: array like, with the numbers of elements that each 1-rank 
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is 
        going to be represented. The value of dim must be coherent with the 
        size of _tshape parameter. \n
        **_rank**: integer type, this variable is created to store the number  
        of modes of the solution.
    **Methods**
    """
    def __init__(self,_tshape,dim):                                                                  
        TensorDescriptor.__init__(self,_tshape,dim)
        self._rank=0                                      
        self._U=[]
#------------------------------------------------------------------------
    def solution_initialization(self):
        """
        This method serve to initiate a new object
        """
        
        for i in range(self._dim):
            self._U.append(np.zeros(self._tshape[i]))
            self._U[i]=np.array([self._U[i]])
#------------------------------------------------------------------------            
        
    def get_rank(self):
        """
        Method to get   _rank value
        """
        return self._rank

    def set_rank(self):
        """
        Method to set a new _rank value
        """
        self._rank=self._rank+1
        
#------------------------------------------------------------------------

    def get_tshape(self):
        """
        Method to get _tshape value
        """
        return self._tshape

    def set_tshape(self,tshape):
        """
        Method to set a new _tshape value
        """
        self._tshape=tshape
#------------------------------------------------------------------------           
     
    def get_U(self):
        
        """
        get_U is the methode to call and get the variable _U
        """
        return self._U
    def set_U(self,newU):
        
        """
        set_U is the methode to call and get _U
        """
        self._U=newU
        
    def add_enrich(self,R):
        """
        add_enrich is the method to add a new mode in each element of the
        _U list attribute.
        """
        for i in range(self._dim):
                    
                       self._U[i]=np.append(self._U[i],np.array([R[i]]),axis=0)
           
#--------------------------------------------------------------------------
    def save(self, file_name):
        """
        This method has the function to save a Canonical class object as a 
        binary file. \n
        **Parameters**:\n
            Object= A Canonical class object.\n
            file_name= String type. Name of the file that is going to be 
            storage. \n
        **Returns**:\n
            File= Binary file that will reproduce the object class when 
            reloaded.
        """
        if type(file_name)!=str:
            raise ValueError('Variable file_name must be a string')
        pickle_out=open(file_name,"wb")
        pickle.dump(self,pickle_out)
        pickle_out.close()
        
    
#--------------------------------------------------------------------------
 
    def writeU(self):
        """
        The method writeU create  .csv (coma separated values) files with _U
        list elements (
        :math:`x_{u,i}`), one file for each mode. The method will print in 
        screen the confirmation that the file was created.
        """
        for i in range (self._dim):
            aux="U("+str(i)+").csv"
            aux2='readfile.read("'+aux+'")'
            print('--------------------------------------------------------\n')
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
       
        
        
        New_Value=CanonicalForme(C._tshape,C._dim) 
        New_Value._rank=C._rank+object2._rank
        New_Value._d=C._d
        
        for i in range(C._dim-1):
            object2._U[i]=np.multiply(-1,object2._U[i])
     
        for i in range(C._dim-1):
            New_Value._U[i]=np.append(C._U[i],object2._U[i],axis=0)
        
        
        
        return New_Value     

#------------------------------------------------------------------------------
        
    def reconstruction(self):
        """
        This function returns a full format class object from the Canonical 
        object introduced. \n
        """
        tshape=self._tshape
        #tshape=tshape.astype(int)
        dim=len(tshape)
        U=self._U[:]
        aux2=U[0].shape
        rank=aux2[0]

        Resultat=np.zeros(shape=tshape)

        R=[]
        for i in range(rank):
            for j in range(dim):
                r=U[j][i][:]
                R.append(r)
            Resultat=new_iteration_result(R, tshape, Resultat)
            R=[]
        
        return Resultat

def new_iteration_result(R, tshape, Resultat):
     
     NewModeResultat=(np.transpose(np.array([R[0]])).dot(np.array([R[1]])))
     
     if len(R)>2:
         for i in range(2,len(R)):
             NewModeResultat=np.kron(NewModeResultat,R[i])
             
         NewModeResultat=NewModeResultat.reshape(tshape)
     
     Resultat=np.add(Resultat,NewModeResultat)
        
                
     return Resultat 
 
    



    
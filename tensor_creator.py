# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:24:54 2018

@author: Diego Britez
"""

import numpy as np
import CartesianGrid


"""
    funcx: This function will create the tensor that represents the the desired 
    function. The variables has to be named as V[0],V[1]....,V[n]
"""
def funcx(V): 
        return 1/(1+(V[0]*np.e**(V[1])))
        #Fonctions tets à utiliser 
        # x*y                              #fonction test 1
        # 1/(1+(x*y))                      #fonction test 2
        # np.sin(np.sqrt(x**2+y**2))       #fonction test 3
        # np.sqrt(1-x*y)                   #fonction test 4
        # 1/(1+(x*np.e**(y)))              #fonction test 5
"""
This code serve to create a tensor of any dimention from a function. 
In the first part of the code the grid that represent each sub-espace of 
the tensor is created 
"""

class TensorCreator():
    def __init__(self):
        self.tshape=[]
        self.dim=[]
        self.lower_limit=[]
        self.upper_limit=[]
        self.X=[]
        self.Grid=[]
        self.F=[]



    """
    Domaine limits
    lower_limit: is an array with the lower limits of each subspace domaine. 
    The number of elements must coincide with the  number of subspaces of the
     problem.It must be introduced by the user as the initialisation is empty.

    upper_limit: is an array with the upper limits of each subspace domaine.
    The number of elements must coincide with the number of subspace of the 
    problem. It must be introduced by the user as  the initialisation is empty.
    """        
    def _dim(self):
        if type(self.lower_limit) != np.ndarray :
            print('lower_limit input must be numpy.array type')
            if type(self.tshape) != np.ndarray :
                
                print('tshape input must be numpy.array type')
                
       
            if type(self.upper_limit) != np.ndarray :
                print('lower_limit input must be numpy.array type')
            if self.lower_limit.size != self.upper_limit.size:
                print("The number of dimetions used in lower and upper limits \
                      are not equals")
                       
        self.dim=self.lower_limit.size
        
    """
    tshape=is an array that contains the information of the number of divitions
    of each sub-space. The number of elements have to coincide with the number
    of subspaces. 
    """
    
    def _CartesianGrid(self):
        
        """
        _CartesianGrid: is the mode that calls the class with the same name.
        CartesianGrid: is the class where all the variables that define 
        the space and the vector is storage. 
        SpaceCreator: is a mode in CartesianGrid class that creates the vectors 
        hat represent the discretized domaine.
        """
        
        self._dim()
        Vector = CartesianGrid.CartesianGrid(self.dim, self.tshape, self.lower_limit,
                                             self.upper_limit)
        aux=Vector.SpaceCreator()
        
        self.X = aux  
        
        
    def _meshgrid(self):
        """
        meshgrid: this function handles the vectors that represent the grid 
        in order to obtain the tensor by one matricial operation. 
        """
        dimention=self.dim
        shape=[]
        limit2=dimention-1
        for i in range(dimention):
            if i==0:
                shape.append(dimention)
            shape.append(np.size(self.X[i]))                   
            self.Grid=np.ones(shape=shape)
    
    
        for i in range(dimention):
            Aux=self.Grid[i]
            
            Aux=np.swapaxes(Aux,i,limit2)
            Aux=np.multiply(Aux,self.X[i])
            Aux=np.swapaxes(Aux,i,limit2)
            self.Grid[i]=Aux
            
    def _Generator(self):
        self._CartesianGrid()
        self._meshgrid()
        self.F=funcx(self.Grid)
        
        return self.X,self.F
    
    def help():
        print("HELP")
        print("====")
        print("First one object class has to be created")
        print("EX: Function1=TensorCreator()")
        print("The class initiates with empty values, so the values of tshape,\
              lower_limit,")
        print("upper_limit are expected as an input (input as numpy.ndarray \
                                                     type expected)")
        print("Function1.tshape=np.array([56,38])")
        print("Function1.lower_limit=np.array([0,0])")
        print("Function1.upper_limit=np.array([1,1])")
        print("After this three elements of the objects were introduced the\
              element dim is")
        print("going to be created inside Grid function, this mode is also\
              used to evaluate")
        print("the coherence and type of data introduced in the first part.")
        print("So once all this data where introduced the mode Generator()\
              has to be called.")
        print("This  mode use the function in the funcx function to generate\
              te Tensor.")
        print("EX:F=Function1.Generator()")
        
        
        
        
        
"""
HELP
====

First one object class has to be created
EX: Function1=TensorCreator1()

The class initiates with empty values, so the values of tshape, lower_limit,
upper_limit are expected as an input (input as numpy.ndarray type expected)

Function1.tshape=np.array([56,38])
Function1.lower_limit=np.array([0,0])
Function1.upper_limit=np.array([1,1])

After this three elements of the objects were introduced the element dim is 
going to be created inside Grid function, this mode is also used to evaluate 
the coherence and type of data introduced in the first part. 

So once all this data where introduced the mode Generator() has to be called.
This  mode use the function in the funcx function to generate te Tensor.

EX:F=Function1.Generator()

"""
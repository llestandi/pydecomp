# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:24:54 2018

@author: Diego Britez
"""

import numpy as np
import CartesianGrid

"""
This code serve to create a tensor with its grid of any dimention from 
a function. 
Parameters:
    - The equation of the function that is going to be found in the first 
    function of this code. The variables must be introduced as 
    V[0],V[1]....,V[n].
    - tshape: will be a list or an ndarray with the information of how many
    values ar going to be taken in each dimention for the discretization 
    process.
    - lower_limit: The values of the inferior limits of the domain for the 
    discretization process.
    - upper_limit: The values of the superior limits of the domain for the 
    discretization process.
Returns:
    Tensor creator will return de grid that represent the domaine and the 
    desired tensor.

For more information how to use this code just type TensorCreator.help()
      
"""

def funcx(V): 
        return np.sin(np.sqrt(V[0]**2+V[1]**2+V[2]**2))
        #Fonctions tets à utiliser 
        # V[0]*V[1]                              #fonction test 1
        # 1/(1+(V[0]*V[1]))                      #fonction test 2
        # np.sin(np.sqrt(V[0]**2+V[1]**2))       #fonction test 3
        # np.sqrt(1-V[0]*V[1])                   #fonction test 4
        # 1/(1+(V[0]*np.e**(V[1])))              #fonction test 5
       
        
        #3D Functions
        #1/(1+V[0]**2+V[1]**2+V[2]**2)
        #np.sin(np.sqrt(V[0]**2+V[1]**2+V[2]**2))
        #V[0]*V[1]*V[2]
        
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
        if type(self.lower_limit)==list:
            self.lower_limit=np.asarray(self.lower_limit)
            
        if type(self.lower_limit) != np.ndarray :
            print('lower_limit input must be numpy.ndarray or list type')
        
        if type(self.tshape) == list:
            self.tshape=np.asarray(self.tshape)
        if type(self.tshape) != np.ndarray :
            print('tshape input must be numpy.ndarray or list type')
                
        if type(self.upper_limit) ==  list:
            self.upper_limit=np.asarray(self.upper_limit)
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
        the space grid is storage. 
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
        print(a)
    
    def _Generator2(self, equation):
        self._CartesianGrid()
        self._meshgrid()
        self.F=func2(self.Grid,equation)
        return self.X, self.F
    
def func2(V,equation): 
    return eval(equation)   
        
        
        
a="""        

HELP
====

First one object class has to be created

EX: Function1=TensorCreator()

The class initiates with empty values, so the values of tshape, lower_limit,
upper_limit are expected as an input (input as numpy.ndarray or list
type expected)

Function1.tshape=[56,38]

Function1.lower_limit=[0,0]

Function1.upper_limit=[1,1]

After this three elements of the objects were introduced the element dim is 
going to be created inside Grid function, this mode is also used to evaluate 
the coherence and type of data introduced in the first part. 

So, once all this data where introduced the mode Generator() has to be called.
This  mode use the function in the funcx function to generate te Tensor.

EX:X, F=Function1._Generator()

where X is a list containing the grid in each dimention and
F is the Tensor created.
"""
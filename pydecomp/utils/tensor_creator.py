# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:24:54 2018

@author: Diego Britez
"""

import numpy as np
import utils.CartesianGrid as cg

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

def funcx(V,case):
    if case==1:
        return 1/(1+np.sum([v**2 for v in V])) #1/(1+x1^2+x2^2...)
    elif case==2:
        return np.sin(np.sqrt(np.sum([v**2 for v in V])))
    elif case==3:
        return 1/(1+np.prod(V,0))
    else:
        raise AttributeError("Case {} is not in list".format(case))
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
# @Diego Okay, why not use a class. Please cleanup and document !!!!!
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

        """

        self.dim=len(self.lower_limit)
        aux=cg.GridCreator(self.lower_limit,self.upper_limit,self.tshape,self.dim)
        self.X = aux.X


    def _meshgrid(self):
        """
        meshgrid: this function handles the vectors that represent the grid
        in order to obtain the tensor by one matricial operation.
        """
        dim=self.dim
        shape=[]
        limit2=dim-1
        for i in range(dim):
            if i==0:
                shape.append(dim)
            shape.append(np.size(self.X[i]))
            self.Grid=np.ones(shape=shape)


        for i in range(dim):
            Aux=self.Grid[i]

            Aux=np.swapaxes(Aux,i,limit2)
            Aux=np.multiply(Aux,self.X[i])
            Aux=np.swapaxes(Aux,i,limit2)
            self.Grid[i]=Aux

    def _Generator(self,num_f):
        self._CartesianGrid()
        self._meshgrid()
        self.F=funcx(self.Grid,num_f)

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

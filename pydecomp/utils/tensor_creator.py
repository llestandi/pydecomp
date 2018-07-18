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

def funcx(X,case):
    d=X.shape[0]
    if case==1:
        return 1/(1+np.sum([X[i]**2 for i in range(d)],axis=0))
    elif case==2:
        return np.sin(np.sqrt(np.sum([X[i]**2 for i in range(d)],axis=0)))
    elif case==3:
        return 1/(1+np.prod(X,0))
    elif case=='Vega' and d==5:
        return X[0]**2*(np.sin(5*X[1]*np.pi + 3*np.log(X[0]**3+X[1]**2+X[3]**3+ \
                X[2]+np.pi**2))-1)**2 +(X[0]+X[2]-1)*(2*X[1]-X[2])*(4*X[4]-X[3])* \
                np.cos(30*(X[0]+X[2]+X[3]+X[4])) * np.log(6+X[0]**2*X[1]**2+X[2]**3) \
                -4*X[0]**2*X[1]*X[4]**3*(-X[2]+1)**1.5
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
        print(self.Grid.shape)
        self.F=funcx(self.Grid,num_f)

        return self.X,self.F

    def help():
        print(a)

    def _Generator2(self, equation):
        self._CartesianGrid()
        self._meshgrid()
        self.F=func2(self.Grid,equation)
        return self.X, self.F

    def __str__(self):
        str= "Shape: {} \n".format(self.tshape)
        str+="dim: {} \n".format( self.dim)
        str+="lower limit: {} \n".format( self.lower_limit)
        str+="upper limit: {} \n".format( self.upper_limit)
        str+="X: {} \n".format( self.X)
        str+="Grid: {} \n".format( self.Grid)
        str+="F: {} \n".format( self.F)
        return str

def func2(V,equation):
    return eval(equation)

#------------------------------------------------------------------------------
def testg(test_function, shape, dim, domain ):
    # @Diego FIXME
    """
    This function propose several models of function to be chosen to build a
    multivariable tensor (synthetic data) , the output
    depends only on the function selected (1,2..) and the number of dimention
    to work with.\n
    **Parameters**\n
    test_function: integer type, describes the format of the function selected:
    \n
    Formule 1 \n
    :math:`1\(1+X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2})`
    \n
    Formule 2 \
    :math:`1\(1+X_{1}^{2}X_{2}^{2}...X_{n}^{2})`
    \n
    Formule 3 \n
    :math:`\sin(\sqrt {X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2}})`
    \n
    **shape**: array or list type number of elements to be taken
    in each subspace. \n
    **dim**:Integer type. Number of dimentions.
    """
    Function=TensorCreator()
    #Creating the variables required for TensorCreator from the data
    Function.lower_limit=np.ones(dim)*domain[0]
    Function.upper_limit=np.ones(dim)*domain[1]
    Function.tshape=shape
    print(Function)
    X,F= Function._Generator(test_function)
    print("X\n", X,'\n F: \n',F)
    return X,F

def testf(test_function, shape, dim, domain ):
    """
    This function propose several models of function to be chosen to build a
    multivariable tensor (synthetic data) , the output
    depends only on the function selected (1,2..) and the number of dimention
    to work with.\n
    **Parameters**\n
    test_function: integer type, describes the format of the function selected:
    \n
    Formule 1 \n
    :math:`1\(1+X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2})`
    \n
    Formule 2 \n
    :math:`\sin(\sqrt {X_{1}^{2}+X_{2}^{2}+...+X_{n}^{2}})`
    \n
    Formule 3 \
    :math:`X_{1}*X_{2}*...*X_{n}`
    \n
    **shape**: array or list type number of elements to be taken
    in each subspace. \n
    **dim**:Integer type. Number of dimentions.
    """
    test_function_possibilities=[1,2,3]
    if test_function not in test_function_possibilities:
        note="""
        Only 3 multivariable test functions are defined, please introduce
        introduce a valid value.
        """
        raise ValueError(note)
    if test_function==1:
        equation='(1'
        for i in range(dim):
            if i<dim-1:
               aux='+V['+str(i)+']**2'
               equation=equation+aux
            else:
               aux='+V['+str(i)+']**2)'
               equation=equation+aux
        equation='1/'+equation
    elif test_function==2:
        equation='np.sin(np.sqrt('
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']**2+'
                equation=equation+aux
            else:
                aux='V['+str(i)+']**2)'
                equation=equation+aux
        equation=equation+')'
    elif test_function==3:
        equation=''
        for i in range(dim):
            if i<dim-1:
                aux='V['+str(i)+']*'
                equation=equation+aux
            else:
                aux='V['+str(i)+']'
                equation=equation+aux


    Function=TensorCreator()
    #Creating the variables required for TensorCreator from the data
    lowerlimit=np.ones(dim)
    lowerlimit=lowerlimit*domain[0]
    upperlimit=np.ones(dim)
    upperlimit=upperlimit*domain[1]
    Function.lower_limit=lowerlimit
    Function.upper_limit=upperlimit
    Function.tshape=shape
    print(Function)
    X,F= Function._Generator2(equation)

    return X,F
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

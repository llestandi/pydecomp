# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:17:19 2018

@author: Diego Britez
"""

import numpy as np
import scipy
from scipy.sparse import diags
# @Diego I have merged several files here, they all relate to the mass matrix.
# However you have made classes that are not all necessary. I think we (you)
# should create a class for mass matrices. Indeed, they have a particular behaviour,
# it is handy to have them always together and you could also add two cases inside
# them that would be transparent both for the user and the routines that need them.
# Basically it contains two cases : one is filled with vectors (very efficient and
# easy to manipulate) while the other one is filled with diagonal matrices. This allows
# more efficient computing as long as you define the required operators (mainly matmul)

# @Diego, do we need a class for that? I don't think so.
"""
In this class we define the grid that will contain the integration points to
integrate with the trapezes  Method.
"""
class IntegrationGrid:

    def __init__(self,X,dim, tshape):
        self.X = X
        self.Xgrid = []
        self.dim = dim
        self.tshape = tshape

    def grille(self,x,nx):
        w=np.zeros(nx)
        w[1:-1]=(x[2:]-x[0:-2])/2
        w[0]=(x[1]-x[0])/2
        w[-1]=(x[-1]-x[-2])/2
        return w

    def IntegrationGridCreation(self):
        for i in range(self.dim):
            w=self.grille(self.X[i],self.tshape[i])
            self.Xgrid.append(w)
        return self.Xgrid


def matricize_mass_matrix(dim,i,M):
    """
    Returns the equivalent mass matrices of a matricized tensor.\n
    **Parameters** \n
    dim= number of dimentions of the problem. \n
    i = dimension number to arange the function. \n
    M= list of sparse.diag matrices as elements type with the mass integration
    (trapeze  method integration points) for each dimention.\n
    **Return** \n
    :math:`M_{x}` =
    The mass matrix as sparse.diag matrix with the integration
    points of the *"i"* dimention.\n
    :math:`M_{t}` =
    The equivalent mass matrix as sparse.diag matrix of n-1 dimentions.\n

    **Definition**\n
    Be
    :math:`M=(M_{1},M_{2}...,M_{i},..,M_{d})` a list with
    :math:`M_{i}`
    the mass sparse.diag matrix of each dimention.\n
    Be *"i"* the dimention selected dimention to rearange the mass list\n
    The result of the function will return:\n
    :math:`M_{x}=M_{i}`\n
    :math:`M_{t}=M_{1} \otimes M_{2} \otimes...M_{i-1} \otimes M_{i+1} \otimes..
    \otimes M_{d}`
    """
    #copy the list of mass matrices to M2
    M2=M[:]
    #moving the actual indice to the first order
    M2.insert(0,M2.pop(i))

    Mt=M2[1]
    Mx=M2[0]
    var=2
    # @Diego I think you can remove this loop using the automatic chaining of Kronecker operator (please check it)
    while var<=dim-1:
        Mt=scipy.sparse.kron(Mt,M2[var],format='dia')
        var=var+1
    return Mx,Mt

def mass_matrices_creator(X):
    """
    Returns a list of   sparse diagonals matrices with the integration points
    for the trapezoidal integration method (mass matrices) from a list of
    cartesian grid points.\n

    **Parameter**\n
    X: list of Cartesian grids vectors. \n

    **Return**
    M:list of mass matrices (integration points for trapeze integration method)
    as sparse.diag type elements. \n

    **Example**
    import high_order_decomposition_method_functions as hf \n
    import numpy as np \n
    X1=np.array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ]) \n
    X2=np.array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ]) \n
    X3=np.array([ 0.        ,  0.33333333,  0.66666667,  1.        ]) \n
    X=[] \n
    X.append(X1) \n
    X.append(X2) \n
    X.append(X3) \n
    M=hf.mass_matrices_creator(X) \n
    print(M) \n
        [<5x5 sparse matrix of type '<class 'numpy.float64'>' \n
        with 5 stored elements (1 diagonals) in DIAgonal format>, \n
        <6x6 sparse matrix of type '<class 'numpy.float64'>' \n
        with 6 stored elements (1 diagonals) in DIAgonal format>, \n
        <4x4 sparse matrix of type '<class 'numpy.float64'>' \n
        with 4 stored elements (1 diagonals) in DIAgonal format>] \n
    """

    dim=len(X)
    tshape=[]

    a="""X's elements must be either numpy.array or list type elements"""
    b="""X's elements must be vectors"""

    for i in range(dim):
        if type(X[i])==list:
            X[i]=np.array(X[i])
        aux=np.array([0])
        if type(X[i])!= type(aux):
            raise ValueError(a)
        if (len(X[i].shape)>1):
            raise ValueError(b)

        aux=X[i].size
        tshape.append(aux)

    M=IntegrationGrid(X,dim,tshape)
    M=M.IntegrationGridCreation()
    for i in range (dim):
        M[i]=diags(M[i])

    return M

def unit_mass_matrix_creator(Data):
    """
    This function will unitarys sparse mass matrix for a tensor, this is going
    to be called when an POD method is wanted to be used as SVD.
    """
    datashape=Data.shape
    mass=[]
    for i in range(len(datashape)):
        massi=np.ones(datashape[i])
        massi=diags(massi)
        mass.append(massi)
    return mass

if __name__=="__main__":
    #X=[]
    #x1=np.linspace(0,1,3)
    #x2=np.linspace(0,1,5)
    #X.append(x1)
    #X.append(x2)
    #tshape=np.array([3,5])
    Xgrid=IntegrationGrid(X,np.size(tshape),tshape)
    Xgrid=Xgrid.IntegrationGridCreation()
    print(Xgrid)
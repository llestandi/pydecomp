# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:17:19 2018

@author: Diego Britez
"""

import numpy as np
from scipy.sparse import diags
import scipy.sparse
import sys
import utils.integrationpoints as ip

class DiaMatrix:
    """
    This class is created to simplify operations with mass matrices. Mass
    matrices can be defined either as a sparse matrix, as a list or as
    a numpy.ndarray.
    This class will unify the matmul operation for any of this formats.
    """
    def __init__(self,M):
        self.M=M
    def __matmul__(self,Matrix):
        if type(self.M)==scipy.sparse.dia.dia_matrix:
            result=self.M@Matrix
        elif (type(self.M)==np.ndarray):
            result=Matrix*self.M[:,np.newaxis]
        elif type(self.M)==list:
            M1=np.array(self.M)
            result=Matrix*M1[:,np.newaxis]
        return result
    def inv(self):
        """
        This fonction will return the inverse of mass matrix as an DiaMatrix
        object
        """
        #possible_mass_matrix_format=[list, scipy.sparse.dia.dia_matrix,
                                # numpy.ndarray]

        list_or_array=[list, np.ndarray]

        if type(self.M)==scipy.sparse.dia.dia_matrix:
            Maux=self.M
            Maux=Maux.diagonal()
            inv=1/Maux
            inv=diags(inv)
            inv=DiaMatrix(inv)
        if type(self.M) in list_or_array:
            inv=1/self.M
            inv=DiaMatrix(inv)
        return inv

    def sqrt(self):
        """
        Returns the square root of the input mass matrix
        """
        reponse=np.sqrt(self.M)
        reponse=DiaMatrix(reponse)
        return reponse

    def transpose(self):
        """
        Returns the transpose of a MassMatrix as an MassMatrix object
        """
        reponse=DiaMatrix(self.M.T)
        return reponse
#------------------------------------------------------------------------------

class MassMatrices:
    """
    Parameters: \n
    List of DiaMatrix objects.
    """
    def __init__(self,DiaMatrix_list):
        self.DiaMatrix_list=DiaMatrix_list
        self.shape=[x.size for x in DiaMatrix_list]


#------------------------------------------------------------------------------


def mass_matrices_creator(X, sparse=False, integration_method='trapeze'):
    """
    Returns a list of   sparse diagonals matrices with the integration points
    for the trapezoidal integration method (mass matrices) from a list of
    cartesian grid points.\n

    **Parameter**\n
    X: list of Cartesian grids vectors. \n
    sparse: if False, the data is going to be expresed as ndarray vector if
    True as sparse vectors.
    integration_method: Default method 'trapeze'. Simpson option is not yet
    avaible.
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

    M=ip.IntegrationPoints(X,dim,tshape)
    M=M.IntegrationPointsCreation()
    if sparse:
        for i in range (dim):
            M[i]=diags(M[i])
    return M

def matricize_mass_matrix(dim,i,M,sparse=False):
    """
    Returns the equivalent mass matrices of a matricized tensor.\n
    **Parameters** \n
    dim= number of dimentions of the problem. \n
    i = dimension number to arange the function. \n
    M= MassMatrices object element.
    **Return** \n
    :math:`M_{x}` =
    The mass matrix  of the *"i"* dimention.\n
    :math:`M_{t}` =
    The equivalent mass matrix  of n-1 dimentions.\n

    **Definition**\n
    Be
    :math:`M=(M_{1},M_{2}...,M_{i},..,M_{d})` a list with
    :math:`M_{i}`
    the  matrix of each dimention.\n
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

    if sparse:
        var=2
        while var<=dim-1:
            Mt=scipy.sparse.kron(Mt,M2[var],format='dia')
            var=var+1
    if not sparse:
        var=2
        while var<=dim-1:
            Mt=scipy.kron(Mt,M2[var])
            var=var+1
        return Mx,Mt

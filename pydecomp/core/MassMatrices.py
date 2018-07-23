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
        type_msg="Wrong type for M : {}".format(type(M))
        if type(M)==scipy.sparse.dia.dia_matrix:
            self.is_sparse=True
            self.shape=M.shape
            self.dia_size=M.shape[0]
        elif type(M)==np.ndarray:
            if M.ndim==1:
                self.is_sparse=False
                self.shape=(M.size,M.size)
                self.dia_size=M.size
            else:
                raise TypeError(type_msg)
        else:
            raise TypeError(type_msg)
        self.M=M

    def __matmul__(self,Matrix):
        if self.is_sparse:
            return self.M@Matrix
        else:
            return Matrix*self.M[:,np.newaxis]

    def inv(self):
        """
        This fonction will return the inverse of mass matrix as an DiaMatrix
        object
        """
        if self.is_sparse:
            Maux=self.M
            Maux=Maux.diagonal()
            inv=1/Maux
            inv=diags(inv)
            inv=DiaMatrix(inv)
        else:
            inv=1/self.M
            inv=DiaMatrix(inv)
        return inv

    def sqrt(self):
        """  Returns the square root of the input mass matrix  """
        return DiaMatrix(np.sqrt(self.M))

    def transpose(self):
        """ Returns the transpose of a MassMatrix as an MassMatrix object """
        return DiaMatrix(self.M.T)

    def __str__(self):
        str="Diamatrix\n"
        str+="is_sparse: {} \n".format(self.is_sparse)
        str+="of shape : {}\n".format(self.shape)
        str+="and size : {}\n".format(self.dia_size)
        str+="M :\n {}".format(self.M)
        return str

class MassMatrices:
    """
    Parameters: \n
    List of DiaMatrix objects.
    """
    def __init__(self,DiaMatrix_list):
        if all(mat.is_sparse for mat in DiaMatrix_list):
            self.is_sparse=True
        elif all(not mat.is_sparse for mat in DiaMatrix_list):
            self.is_sparse=False
        else:
            raise NotImplementedError("All mass matrices need to be of the same sparseness")
        self.Mat_list=DiaMatrix_list
        self.shape=[x.dia_size for x in DiaMatrix_list]
        self.ndim=len(self.shape)

    def update_mass_matrix(self,id,M_new):
        """Allows the user to change mass matrix M[id] with M_new"""
        if M_new.is_sparse!=self.is_sparse:
            raise AttributeError("self and M_new are of the same sparseness")
        self.shape[id]=M_new.shape
        self.Mat_list[id]=M_new

    def pop(self,i):
        """Removes item i from MassMatrices """
        self.Mat_list.pop(i)
        self.shape.pop(i)

    def __len__(self):
        return len(self.Mat_list)

    def __str__(self):
        string="----------------------------------\n"
        string+="Mass matrices list"
        string+="\n is_sparse={}".format(self.is_sparse)
        string+="\n Of shape: {}".format(self.shape)
        for i in range(len(self.shape)):
            string+= "\n"+str(self.Mat_list[i])
        string+="\n----------------------------------\n"
        return string

    # def __iter__(self):
    #     return(self.Mat_list)


def pop_1_MM(MM):
    """ Removes the first item of MM and returns a new mass matrices """
    return MassMatrices(MM.Mat_list[1:])

def identity_mass_matrix(size,sparse=False):
    """ Builds the identity mass matrix """
    if sparse==True:
        return diag(np.ones(size))
    else:
        return DiaMatrix(np.ones(size))


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
    for i in range (dim):
        if sparse:
            M[i]=diags(M[i])
        M[i]=DiaMatrix(M[i])
    M=MassMatrices(M)
    return M

def matricize_mass_matrix(i,M):
    """
    Returns the equivalent mass matrices of a matricized tensor.\n
    **Parameters** \n
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
    if type(M) != MassMatrices:
        raise AttributeError('M must be a MassMatrices')
    #copy the list of mass matrices to M2
    M2=M.Mat_list[:]
    #moving the actual indice to the first order
    dim=M.ndim
    print(M)
    M2.insert(0,M2.pop(i))
    Mx=M2[0].M
    Mt=M2[1].M
    if M.is_sparse:
        for i in range (2,dim):
            Mt=scipy.sparse.kron(Mt,M2[i],format='dia')
    else :
        for i in range (2,dim):
            Mt=np.kron(Mt,M2[i].M)
    return Mx,Mt

def matricize_mass_matrix_for_TT(M,is_first=True):
    """
    Returns the equivalent mass matrices of a matricized tensor, special case for
    TT-SVD.\n
    **Parameters** \n
    dim= number of dimentions of the problem. \n
    M= MassMatrices object element.
    is_first= true for first matricizatin, then simple call to the usual matricize_mass_matrix
            false then actually do the special matricization

    **Return** \n
    :math:`M_{x}` =
    The mass matrix  of the *"i"* dimention.\n
    :math:`M_{t}` =
    The equivalent mass matrix  of n-1 dimentions.\n

    """
    if type(M) != MassMatrices:
        raise AttributeError('M must be a MassMatrices')
    d=M.ndim
    if is_first:
        return matricize_mass_matrix(0, M)
    else:
        M1=MassMatrices(M.Mat_list[:1])
        M2=MassMatrices(M.Mat_list[2:])
        Mx=Kronecker(M1)
        Mt=Kronecker(M2)


        # M2=M.Mat_list[:]
        # Mt=M2[2].M
        # if M.is_sparse:
        #     Mx=scipy.sparse.kron(M2[0].M,M2[1].M,format='dia')
        #     for i in range (2,d):
        #         Mt=scipy.sparse.kron(Mt,M2[i],format='dia')
        # else :
        #     Mx=np.kron(M2[0].M,M2[1].M)
        #     for i in range (2,d):
        #         Mt=np.kron(Mt,M2[i].M)

        return Mx,Mt

def Kronecker(MM) :
    """ Kronecker product of each matrices in massmatrix structure. """
    M=MM.Mat_list[0].M
    if MM.is_sparse:
        for i in range (1,len(MM)):
            M=scipy.sparse.kron(M,MM.Mat_list[i].M,format='dia')
    else :
        for i in range (1,len(MM)):
            M=np.kron(M,MM.Mat_list[i].M)
    return DiaMatrix(M)

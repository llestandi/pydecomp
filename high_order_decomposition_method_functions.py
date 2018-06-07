# -*- coding: utf-8 -*-
"""
Created on Thu May  3 10:30:18 2018

@author: Diego Britez

"""

import numpy as np
import scipy.sparse
import operator
from scipy.sparse import diags
import integrationgrid
import pickle

def multilinear_multiplication(phi,F,dim):
    """
    **Parameters**:_ \n
    
    phi= Is a list with n-array type elements as the elements.\n \n
    F  = 2d Array type (unfolded Tensor expresed as a matrix). \n
    dim = tuple, list or array type that its value must be coherent with  
    the number of elements of phi. \n \n
    **Return**:_ \n
    
    W= Array type element, result of an unfolded multilinear multiplication. \n
    
    **Definition** \n
    In general the *unfolding multilinear multiplication* is given by: \n
    :math:`[(\phi_{1},\phi_{2},...,\phi_{n}).F]^{n}=\phi_{n}F^{n}
    (\phi_{1} \otimes ... \otimes \phi_{m-1} \otimes \phi_{m+1} \otimes ... 
    \otimes \phi_{d})^{T}` \n
    where
    :math:`\otimes` represents the Kronecker product.
        
    
    """
    W=F
    formeW=[x for x in F.shape]
    for i in range(dim):
       W=matricize(W,dim,i)
       W=phi[i]@W
       actual_dimention=[phi[i].shape[0]]
       del formeW[i]
       actual_dimention.extend(formeW)
       formeW=actual_dimention
       W=np.reshape(W,formeW)
    return W.T
""" 
#optional algorithm for multilinear_multiplication
def multilinear_multiplication(PHI, F, dim):
    index=finding_biggest_mode(PHI)
    print(index)
    PHI2=PHI[:]
    PHI2=rearrange(PHI2,index)
    Fmat=matricize(F,dim,index)
    aux=PHI2[1]
    for i in range(1,dim-1):
        aux=np.kron(aux,PHI2[i+1])
    aux=aux.T
    
    MnX=PHI2[0]@Fmat
    W=MnX@aux
    forme_W=new_forme_W(PHI2)
    W = W.reshape(forme_W)
    W  =final_arrange(W,index)
    return W

"""

#------------------------------------------------------------------------------
def matricize(F,dim,i):
    """
    Returns the tensor F rearanged as a matrix with the *"i"* dimention as 
    the new principal (order 0) order. In other words it returns the unfolded 
    tensor as a matrix with the "i" dimention as the new "0" dimention.  \n 
    **Parameters**\n
    F= array type of n dimentions.\n
    dim= number of dimentions of the tensor F.\n
    i= dimention that is going to be taken as the principal to matricize.
    """
   
    aux=1
    F1=np.moveaxis(F,i,0)
    lista1=F1.shape
    for j in range(1,dim):
        #aux will return the multiplication of dim-1 elements of F1.shape
        
        aux=aux*lista1[j]
    F1=F1.reshape((lista1[0],aux))
    return F1
#------------------------------------------------------------------------------
    
def  matricize_mass_matrix(dim,i,M):
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
    while var<=dim-1:
        Mt=scipy.sparse.kron(Mt,M2[var],format='dia')
        var=var+1
    return Mx,Mt
#------------------------------------------------------------------------------        
def finding_biggest_mode(PHI):
    """
    Returns the index of the maximal value in a list
    """
    
    PHI_shape=[np.size(i) for i in PHI]
    index, value = max(enumerate(PHI_shape), key=operator.itemgetter(1))
    
    return index 
#------------------------------------------------------------------------------

def rearrange(PHI,index):
     """
     Given a list *PHI* and an *index* this function will return a new list
     with the "index" element of the list as the first element.
     """
     if index > len(PHI):
         raise ValueError("index is higher than number of dimentions");
     PHI.insert(0,PHI.pop(index))
     return PHI
 
#------------------------------------------------------------------------------
          
    
def list_transpose(phi):
    """
    Parameter:
        phi: list with n-array matrices as elements.
        
    Return:
        phit: list with n-array transposed matrices of input parameter.
    """
    phit=[i.T for i in phi]
    return phit


#------------------------------------------------------------------------------
def new_forme_W(PHI):
    """
    This function returns the shape as list of the core matrix in a tucker
    type decomposition.
    """
    forme_W=[i.shape[0] for i in PHI]
    
    return forme_W
#------------------------------------------------------------------------------
def final_arrange(W,index):
    """
    Returns an array with the *"index"* dimention as the firs dimention of 
    the array.
    """
    W=np.moveaxis(W,0,index)
    
    return W 
#------------------------------------------------------------------------------
def integrationphi(PHIT,M):
    a="""
    Dimentions of decomposition values list and Mass matrices list are not
    coherents
    """
    if len(PHIT)!= len(M):
        raise ValueError(print(a))
    integrated_phi=[phit@m for (phit,m) in zip(PHIT,M)]
    
    return integrated_phi
#------------------------------------------------------------------------------
    
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
    M=integrationgrid.IntegrationGrid(X,dim,tshape)
    M=M.IntegrationGridCreation()
    for i in range (dim):
        M[i]=diags(M[i])    
 
    
    return M
#------------------------------------------------------------------------------
    
def orth(dim,Xgrid,R,C): 
    """
    This function serves to verify the orthogonality in L_{2} between the modes
    in the canonical class.
    """
       
    for i in range(dim):
            
            Orth=orthogonality_verification(Xgrid[i],R[i],C._U[i])
            Verification=1
            for j in range(C.get_rank()):
                if (Orth[j]>1e-10):
                    print('------------------------------------------------\n')
                    print('Orthogonality is not verified at mode:', C.get_rank())
                    print('Variable of failure= U(',i,')')
                    print('Value of the scalar product of failure=', Orth[j])
                    Verification=0
    return Verification 

def orthogonality_verification(w,R,RR):
    """
    Auxiliar sub function of orth
    """
    aux=np.transpose(np.multiply(w,RR))
    Orth=np.dot(R,aux)
    return Orth  
#------------------------------------------------------------------------------
    
def load(file_name):
    """
    This function will load a file (variable, classe object, etc) in pickle 
    format in to its python orinal variable format.\n
    **Parameters**:\n
    file_name: string type, containg the name of the file to load.\n
    directory_path= is the adreess of the folder where the desired file is 
    located.
    **Returns**\n
    
    Variable: could be an python variable, class object, list, ndarray 
    contained in the binary file. \n
    
    **Example** \n
    import high_order_decomposition_method_functions as hf  \n

    FF=hf.load('example_file')
    
    """
    
    if type(file_name) != str:
        file_name_error="""
        The parameter file_name must be a string type variable
        """
        raise TypeError(file_name_error)
    file_in=open(file_name,'rb')
    Object=pickle.load(file_in)
    file_in.close()
    return Object
#-----------------------------------------------------------------------------
def save(variable, file_name):
    """
        This function will save a python variable (list, ndarray, classe  
    object, etc)  in a pickle file format .\n
        **Parameters**:\n
            Variable= list, ndarray, class object etc.\n
            file_name= String type. Name of the file that is going to be 
            storage. \n
            directory_path=string type. Is the directory adress if the file 
            is going to be saved in a desired folder.
        **Returns**:\n
            File= Binary file that will reproduce the object class when 
            reloaded. \n
    **Example**\n
    import high_order_decomposition_method_functions as hf

    hf.save(F,'example_file') \n
    
    Binary file saved as'example_file'
    """
   
        
    if type(file_name)!=str:
        raise ValueError('Variable file_name must be a string')
    pickle_out=open(file_name,"wb")
    pickle.dump(variable, pickle_out)
    pickle_out.close()
    print('Binary file saved as'+"'"+file_name+"'")
#-----------------------------------------------------------------------------
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
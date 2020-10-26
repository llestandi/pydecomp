# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:19:09 2018

@author: Diego Britez
"""
import numpy as np
import warnings
from scipy.sparse import diags
from core.tensor_algebra import truncate_modes

def TSVD(F, epsilon = 1e-10, rank=100, solver='EVD'):
    """
    This function calculates a matrix decomposition by using the truncated SVD
    method.\n
    The decomposition of the F matrix has following forme:\n
    :math:`F=U.\sigma.A^{t}` \n
    where U and A are the first and second subspaces projection of the matrix.\n
    :math:`\sigma` a square matrix with the singular values as the diagonal
    elements.\n

    **Parameters**\n
    F= 2 dimention ndarray type of data (Matrix).\n
    epsilon= maximal value for the sigular value. Default value= 1e-10.\n
    rank= integer type. Represents the maximal value of the rank that is
    going to be admited. If this value exceeds the maximal rank in SVD method,
    the epsilon criteria will be applied. \n
    *solver*, str, either "SVD" or "EVD"
    **Returns**\n

    U: 2d array type \n
    :math:`\sigma` : sparse diagonal matrix type.\n
    A: 2d array type.\n


    """
    if solver=='SVD':
        U, S, V = np.linalg.svd(F, full_matrices=True)
        V=V.T
        s,u=truncate_modes(S, epsilon, rank, U)
        s,v=truncate_modes(S, epsilon, rank, V)

    elif solver=='EVD':
        try:
            print("trying SVD by EVD")
            u,s,v = SVD_by_EVD(F,tol=epsilon,rank=rank)
        except:
            print("it failed, trying the direct solver")
            u,s,v= TSVD(F, epsilon, rank, solver='PRIMME')
            print("new singular values :{}".format(s))

    elif solver=='PRIMME':
        print("Selected PRIMME_SVDS solver. This solver is iterative and best\
               suited for sparse tall skinny matrices. High accuracy requirement\
               may lead to intractable CPU times.")
        try :
            import primme
        except:
            print("An error occured importing prime. please make sure the package\
                  is installed. \n \
                  # Install a pip package in the current Jupyter kernel\
                  import sys\
                  !{sys.executable} -m pip install primme")
        print(min(F.shape))
        if rank > 0 :
            k=min(rank,min(F.shape))
        else :
            k=min(F.shape)
        svecs_left, svals, svecs_right =  primme.svds(F, k,which='LM', tol=epsilon)
        u=svecs_left
        s=svals
        v=svecs_right.T
    return u,s,v


def SVD_by_EVD(F,tol=0,rank=-1):
    """ This function returns the SVD by solving the EVD problem on F.T F or F F.T"""

    shape=F.shape
    Transposed_POD=False
    if shape[1]>shape[0]:
        F=F.T
        Transposed_POD=True

    C=F.T@F
    Lambda , U =np.linalg.eigh(C)
    # Reversing order
    Lambda = Lambda[::-1]
    U=U[::,::-1]

    Lambda, U=truncate_modes(Lambda,tol,rank,U)
    sigma=np.sqrt(Lambda)
    if np.isnan(sigma).any():
        print(Lambda)
        print("EVD returns negative number leading to nan in sigma")
        return
    r=sigma.size
    inv_sigma=np.reshape(1/sigma,(r,1))

    phi=F@(inv_sigma*U.T).T

    if Transposed_POD:
        U,phi=phi,U

    return phi, sigma, U


if __name__=="__main__":
    from time import time
    print("\n Testing SVD with random matrix\n")
    n,m=300,50
    F=np.random.rand(n,m)
    # F=np.reshape(np.arange(n*m),(n,m))

    t=time()
    phi, sigma, A=TSVD(F,solver='SVD')
    print('--------------SVD computing time {}',time()-t)
    # print("SVD\n----------- \n phi :\n {}\n sigma:\n {} \n A:\n {}".format(phi, sigma, A))
    r=sigma.size
    sigma=np.reshape(sigma,[r,1])
    F_approx = phi@(sigma*A.T)
    err=np.linalg.norm(F_approx-F)
    print("\n Should be small : {}".format(err))

    t=time()
    phi, sigma, A=TSVD(F,solver='EVD')
    print('\n --------------SVD by EVD computing time {}',time()-t)
    # print("\nEVD\n----------- \nphi :\n {}\n sigma:\n {} \n A:\n {}".format(phi, sigma, A))
    r=sigma.size
    sigma=np.reshape(sigma,[r,1])
    F_approx = phi@(sigma*A.T)
    err=np.linalg.norm(F_approx-F)
    print("\n Should be small : {}".format(err))

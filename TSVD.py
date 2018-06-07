# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:19:09 2018

@author: Diego Britez
"""
import numpy as np
from scipy.sparse import diags
def TSVD(F, epsilon = 1e-10, rank=100):
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
    **Returns**\n
    
    U: 2d array type \n
    :math:`\sigma` : sparse diagonal matrix type.\n
    A: 2d array type.\n
    
    
    """
    U, S, V = np.linalg.svd(F, full_matrices=True)
    
    
    
    i=0
    imax=len(S)
    error=[x**2 for x in S]
    error=sum(error)
    error=np.sqrt(error)
    
    
    #Verification if default value has changed
    if (rank!=100):
        if imax>=rank:
            imax=rank
            s=S[:rank]
            s=diags(s)
            u=U[::,:rank]
            v=V[::,:rank]
        else:
            maximal_rank_message="""
            Input rank is higher than maximal rank possible, epsilon criteria
            is going to be applied for now on.
            """
            print(maximal_rank_message)
            while (error>epsilon) & (i<imax):
                error=S[(i+1):]
                error=[x**2 for x in error]
                error=sum(error)
                error=np.sqrt(error)
                i+=1
                srank=i
            
            s=S[:srank]
            s=diags(s)
            u=U[::,:srank]
            V=V.T
            v=V[::,:srank]
    else:
  
        while (error>epsilon) & (i<imax):
          
            error=S[(i+1):]
            error=[x**2 for x in error]
            error=sum(error)
            error=np.sqrt(error)
            i+=1
            srank=i
            
        s=S[:srank]
        s=diags(s)
        u=U[::,:srank]
        V=V.T
        v=V[::,:srank]
    return u,s,v
    
 
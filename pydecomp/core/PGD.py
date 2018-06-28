# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018
@author: Diego Britez
Major # REVIEW:  on 28/06/18
@author: Lucas Lestandi
"""
import numpy as np
from scipy.linalg import norm
from Canonical import CanonicalFormat

def PGD(M,F, epenri=1e-12, maxfix=15):
    """
    This function use the PGD method for a multivariate problem decomposition,
    returning an Canonical class objet.
    \n
    **Parameters**:\n
    **M**:list of mass matrices (integration points for trapeze integration
    method) as sparse.diag type elements\n
    **F**: ndarray type, it's the tensor that its decomposition is wanted.\n
    **epenri**: Stop criteria value for the enrichment loop, default value=1e-10.
    This value is obtained by the divition  of the first mode with the
    mode obtained in the last fixed-point iteration of the last variable
    (wich carries all the information of the module of the function),  more
    detailed information can be find in  Lucas Lestandi's thesis.

    **maxfix**:is the maximal number of iteration made inside of the fix point
    function before declare that the method has no convergence.
    Default value=15 \n

    **Introduction to the PGD method**\n

    Be :math:`F` the tensor that is going to be decomposed by the PGD method.
    And be
    :math:`u^{p}`
    the solution, so we can express:\n
    :math:`u^{p}=\sum_{m=1}^{p}\prod_{i=1}^{D}X_{i}^{p}(x_{i})`.\n
    The weak  formulation of the problem can be expressed as:\n
    :math:`\int_{\Omega}u^{*}(u-F)=0`, is the test function, \n
    for all :math:`u^{*} \in H^{1}(\Omega)`. If we take;\n
    :math:`u^{*}=\prod_{i=1}^{s-1} X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})`,\n
    the weak formulation becomes:\n
    :math:`\int_{\Omega}[\prod_{i=1}^{s-1}X_{i}^{k+1}(x_{i})X^{*}(x_{s})
    \prod_{i=s+1}^{D}X_{i}^{k}(x_{i})(u^{p-1}+\prod_{i=1}^{s}X_{i}^{k+1}
    (x_{i})\prod_{i=s+1}^{D}X_{i}^{k}(x_{i})-F)]=0` \n
    \n

    that for convenience it could be expressed like; \n
    \n
    :math:`\\alpha^{s}\int_{\Omega_{s}}X^{*}(x_{s})X_{s}^{k+1}(x_{s})dx_{s}=
    -\int_{\Omega_{s}}X^{*}\sum_{j=1}^{p-1}(\\beta^{s}(j)X_{s}^{j})
    dx_{s}+\int_{\Omega_{s}}X^{*}\gamma^{s}(x_{s})dx_{s}`\n
    where
    :math:`\\alpha , \\beta , \gamma` are variables deduced from the precedent
    equation and that are expoded in the fix point variables section.

    """
    tshape=F.shape
    dim=len(tshape)
    #Start value of epn that allows get in to the enrichment loop for the first iteration
    eps=1
    n_iter=0
    M=[x.toarray() for x in M]
    M=[np.diag(x) for x in M]

    C=CanonicalFormat(tshape,dim)
    C.solution_initialization()

    while (eps>=epenri):
        C.set_rank()
        R,n_iter=fixed_point(M,C._tshape,C._U,F,n_iter ,C._rank,maxfix)
        C.add_enrich(R)

        if C.get_rank()==1:
            REF=R[dim-1]

        epn=norm(R[dim-1])/norm(REF)
    # @Diego is this necessary?
    #Eliminating the first (zeros) row created to initiate the algorithm
    C._U=[x[1:,::] for x in C._U]
    return C

def fixed_point(M,tshape,U,F,z,r,max_iter):
    """
    This function calculates the n mode for each iteration in the
    enrichment loop for the PGD method. The definition of each variable
    (such as alpha, betha, etc) that it's used here could be found in detail
    in the manuscript of  Lucas Lestandi doctoral thesis.\n
    **Parameters** \n
    X= Cartesian Grid.\n
    tshape= Tensor shape. \n
    U= (n-1) modes of the decomposition. \n
    F= Tensor to be decomposed. \n
    z= No convergence times counter variable. \n
    r= Actual rank in the decomposition. \n
    maxfix= maximun number of iterations in point fix algorithm.\n
    **Returns** \n
    R = n mode of the decomposition in the PGD decomposition method.\n
    z = actualized number of iteration that did not converge, so: \n
    :math:`z_{n}=z_{n-1}+1` if the actual mode does not converge or \n
    :math:`z_{n}=z_{n-1}`
    if it has converged.
    """
    dim=np.size(tshape)
    # @TODO @Diego Replace this useless call to class, with use of rank ones
    # tensors in canonical format. Much clearer
    New_Solution=IterationSolution(tshape,dim)
    New_Solution._current_solution_initialization()
    R=New_Solution.R


    Old_Solution=New_Solution
    k=0
    eppf=1e-8  # @TODO @Diego Variable naming should say what it does (without comment)
    epsilon=1

    while ((epsilon>=eppf) & (k<max_iter)):
        k=k+1
        Old_Solution=R[dim-1]

        for i in range(dim):
            Alpha=alpha(R,M,i,dim)
            Gamma=gamma(R,F,M,i,dim)
            Betha=betha(M,R,U,i,dim)

            aux=np.dot(U[i].T,Betha)
            aux=np.transpose(aux)
            R[i]=(-aux+Gamma)/Alpha

            if (i<(dim-1)):
                R[i]=R[i]/(norm(R[i]))

        epsilon=norm(R[dim-1]-Old_Solution)/norm(Old_Solution)

        # @TODO @Diego
        # I don't see the point of having this z variable + it prevents more
        # concise writing of enriching Canonical approx
        if k==itmax:
            z=z+1
    return  R,z

# @TODO @Diego
# This class is useless, please remove and rely on rank one tensors in canonical format
class IterationSolution:
    def __init__(self,tshape,dim):
        self.R=[]
        self.tshape=tshape
        self.dim=dim
    def _current_solution_initialization(self):
        for i in range (self.dim):
            self.R.append(np.array([np.ones(self.tshape[i])]))



def alpha(R,M, i, dim):
    """
    # @TODO @Diego This is a mathematical description, please document the function itself too
    This function calculates the value of the
    :math:`\\alpha`
    variable in the fix point algorithm:  \n

    :math:`\\alpha^{s}=\int_{\Omega/\Omega_{s}}\prod_{i=1}^{s-1}(X_{i}^{k+1})^
    {2}\prod_{i=s+1}^{d}(X_{i}^{k})^{2}`, which could be expressed also as: \n
    :math:`\\alpha^{s}=\prod_{i=1}^{s-1}\int_{\Omega_{i}}(X_{i}^{k+1})^
    {2}dx_{i}\prod_{i=s+1}^{d}\int_{\Omega_{i}}(X_{i}^{k})^{2}	dx_{i}`
    """
    R1=R[:]
    alpha=1
    R1[0],R1[i]=R1[i],R1[0]
    M1=M[:]
    M1[0],M1[i]=M1[i],M1[0]

    # @TODO @Diego Thats a weird integral, I think you need to uniformize it
    # with the rest of the code. Remove loop if possible
    for j in range(dim-1,0,-1):
        R1[j]=np.multiply(R1[j],R1[j])
        aux=R1[j]@M1[j]
        alpha=alpha*aux
    return alpha


def gamma(R,F,M,i,dim):
    """
    # @TODO @Diego This is a mathematical description, please document the function itself too
    This function will return the value of the
    :math:`gamma
    variable for each iteration in the fix poitn algorithm.
    :math:`\gamma^{s}(x_{s})= \int_{\Omega/\Omega_{s}}(\prod_{i=1}^{s-1}
    X_{i}^{k+1}\prod_{i=s+1}^{D}X_{i}^{k}).F`
    """
    F2=F
    F2=np.swapaxes(F2,0,i)
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]
    M1=M[:]
    M1[0],M1[i]=M1[i],M1[0]

    #@TODO improve this by removing loop (use Kronecker and matmul)
    for j in range(dim-1,0,-1):

        F2=np.multiply(F2,R1[j])
        F2=F2@M1[j]
    return F2

def beta(M,R,U,i,dim):
    """
    # @TODO @Diego This is a mathematical description, please document the function itself too
    This function calculates the value of the
    :math:`\\beta`
    variable in the fix point algorithm:  \n

    :math:`\\beta^{s}(j)=\int_{\Omega/\Omega_{s}}\prod_{i=1}^{s-1}(X_{i}^{k+1}
    X_{i}^{j})\prod_{i=s+1}^{d}(X_{i}^{k}X_{i}^{j})`, which  could be expressed
    as, \n
    :math:`\\beta^{s}(j)=\prod_{i=1}^{s-1}\int_{\Omega}(X_{i}^{k+1}X_{i}^{j})
    \prod_{i=s+1}^{d}\int_{\Omega_{s}}\prod_{i=s+1}^{d}(X_{i}^{k}
    X_{i}^{j})dx_{i}`
    """
    M1=M[:]
    M1[0],M1[i]=M1[i],M1[0]
    U1=U[:]
    U1[0],U1[i]=U1[i],U1[0]
    R1=R[:]
    R1[0],R1[i]=R1[i],R1[0]

    #@TODO improve this by removing loop (use Kronecker and matmul)
    Betha=1
    for j in range(1,dim):
        aux2=np.multiply(U1[j],R1[j])
        aux2=aux2@M1[j]
        Betha=Betha*aux2
    return Betha

if __name__=="__main__":
    # @TODO @Diego This is a simple copy paste of old files, please write a Simple
    # test for PGD
    tshape=np.array([5,4,8])
    dim=3
    R=IterationSolution(tshape,dim)
    R._current_solution_initialization()
    R=R.R

    F=np.arange(120)
    F=F.reshape(5,6,4)
    lower_limit=np.array([0,0,0])
    upper_limit =np.array([1,1,1])
    tshape=np.array([5,6,4])
    dim=3

    Vector = CartesianGrid.CartesianGrid(dim, tshape, lower_limit, upper_limit)

    X = Vector.SpaceCreator()
    New_Solution=IterationSolution(tshape,dim)
    New_Solution._current_solution_initialization()
    R=New_Solution.R
    i=1

    F2=F
    R2=R
    R2[0],R2[i]=R2[i],R2[0]
    X2=X
    X2[0],X2[i]=X2[i],X2[0]
    F2=np.swapaxes(F2,0,i)


    for j in range(dim-1,1,-1):
        F2=np.multiply(F2,R2[j])
        F2=integration_1dtrap(F2,X2[j])


    #Unmark commentary if Orthogonality Verification is desired
    """
    if (Verification==1):
                print('Orthogonality between modes was verified')
                print('----------------------------------------\n')
    print('--------------------------------------------------------------\n')
    print("Iteration's enrichment loops=",C.get_rank())
    print("Iteration's enrichment loops in fixed-point loop with no convergence:",z)
    print('Epsilon=',epn)
    """

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:43:55 2018

@author: Diego Britez
"""
import numpy as np
import csv
import pickle

from deprecated.tensor_descriptor_class import TensorDescriptor
import pydecomp.core.tensor_algebra as ta
import deprecated.full_format
import timeit
#------------------------------------------------------------------------
# @Diego, class is working but a lot of simplification work is needed with your
# new python knowledge + homogeneisation of tensor format classes.

class CanonicalTensor(TensorDescriptor):
    """
    **Canonical Type Format**


    In this format, any tensor
    :math:`\chi\in V = \otimes_{\mu=1}^{d}V_{\mu}`
    a tensor space, is written as the finite sum of rank-1 tensors.
    :math:`\chi \in C_{r} (\mathbb{R})`
    is said to be represented in the
    canonical format and it reads, \n
    :math:`\chi=\sum_{i=1}^{r}{\otimes}_{\mu=1}^{d}x_{u,i}`
    \n
    **Attributes**

        **_U** : list type, in this list will be stored all the modes of the
        decomposed matrix as an array type each mode (
        :math:`x_{u,i}`).\n
        **_tshape**: array like, with the numbers of elements that each 1-rank
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is
        going to be represented. The value of dim must be coherent with the
        size of _tshape parameter. \n
        **_rank**: integer type, this variable is created to store the number
        of modes of the solution.
    **Methods**
    """
    def __init__(self,_tshape,dim):
        TensorDescriptor.__init__(self,_tshape,dim)
        self._rank=0
        self._U=[]
#------------------------------------------------------------------------
    def solution_initialization(self):
        """
        @Diego, this is redundant with __init__ #FIXME
        This method serve to initiate a new object
        """

        for i in range(self._dim):
            self._U.append(np.zeros(self._tshape[i]))
            self._U[i]=np.array([self._U[i]])
#------------------------------------------------------------------------

    def get_rank(self):
        return self._rank

    def set_rank(self,r):
        self._rank=r

    def rank_increment(self):
        self._rank=self._rank+1

#------------------------------------------------------------------------

    def get_tshape(self):
        return self._tshape

    def set_tshape(self,tshape):
        self._tshape=tshape
#------------------------------------------------------------------------

    def get_U(self):
        return self._U
    def set_U(self,newU):
        self._U=newU

    def add_enrich(self,R):
        """
        add_enrich is the method to add a new mode in each element of the
        _U list attribute.
        """
        self.rank_increment()
        for i in range(self._dim):
            self._U[i]=np.append(self._U[i],R[i],axis=0)

#--------------------------------------------------------------------------
    def save(self, file_name):
        """
        This method has the function to save a Canonical class object as a
        binary file. \n
        **Parameters**:\n
            Object= A Canonical class object.\n
            file_name= String type. Name of the file that is going to be
            storage. \n
        **Returns**:\n
            File= Binary file that will reproduce the object class when
            reloaded.
        """
        if type(file_name)!=str:
            raise ValueError('Variable file_name must be a string')
        pickle_out=open(file_name,"wb")
        pickle.dump(self,pickle_out)
        pickle_out.close()


#--------------------------------------------------------------------------

    def writeU(self):
        """
        The method writeU create  .csv (coma separated values) files with _U
        list elements (
        :math:`x_{u,i}`), one file for each mode. The method will print in
        screen the confirmation that the file was created.
        """
        for i in range (self._dim):
            aux="U("+str(i)+").csv"
            aux2='readfile.read("'+aux+'")'
            print('--------------------------------------------------------\n')
            print('The file', aux , 'has been created,to read it, type:\n',
                    aux2)

            with open(aux,'w',newline='') as fp:
                a=csv.writer(fp, delimiter=',')
                a.writerows(self._U[i])

            with open('tshape.csv','w', newline='') as file:
                b=csv.writer(file, delimiter=',')
                b.writerow(self._tshape)


#----------------------------------------------------------------------------

#Redefining the substractions for objects with Canonical forme

    def __sub__(C,object2):

        if (isinstance(object2,CanonicalTensor)==False):
            print('New object has not the correct canonical forme')

        if (C._d!=object2._d):
            print('Fatal error!, Objects dimentions are not equals!')

        for i in range(C._d):
            if (C._tshape[i]!=object2._tshape[i]):
                print('Fatal error!, shapes of spaces are not compatibles!')



        New_Value=CanonicalTensor(C._tshape,C._dim)
        New_Value._rank=C._rank+object2._rank
        New_Value._d=C._d

        for i in range(C._dim-1):
            object2._U[i]=np.multiply(-1,object2._U[i])

        for i in range(C._dim-1):
            New_Value._U[i]=np.append(C._U[i],object2._U[i],axis=0)



        return New_Value

#------------------------------------------------------------------------------

    def reconstruction(self):
        """
        This function returns a full format class object from the Canonical
        object introduced. \n
        **Deprecated**
        """
        tshape=self._tshape
        #tshape=tshape.astype(int)
        dim=len(tshape)
        U=self._U[:]
        aux2=U[0].shape
        rank=aux2[0]

        Resultat=np.zeros(shape=tshape)

        R=[]
        for i in range(rank):
            for j in range(dim):
                r=U[j][i][:]
                R.append(r)
            Resultat=new_iteration_result(R, tshape, Resultat)
            R=[]
        return Resultat

    def to_full_quick(self,rank=-1):
        """
        @author : Lucas 27/06/18
        Provides a quick evaluation (based on kathri rao product) of rank r
        truncated self.
        It is based on Kolda formula for evaluation of Canonical format in matrix
        formulation. (much more efficient for large arrays, still not as fast as
        Kosambi version which is using eigsum)

        **Parameters**:
        *self* To be evaluated
        *rank* [int] Truncation rank, if out of range, set to maximum value

        **Return** ndarray of shape = self._tshape
        """
        if rank < 0 or rank > self.get_rank():
            r=self.get_rank()
        else:
            r=rank+1
        U_trunc=[(np.stack(self._U[i][:r])).T for i in range(self._dim)]
        shape=self._tshape
        flat_eval=U_trunc[0] @ np.transpose(ta.multi_kathri_rao(U_trunc[1:]))
        return np.reshape(flat_eval,shape)


    def memory_eval(self,r):
        """
        @author : Lucas 27/06/18
        Returns the number of floats required to store self at rank r
        """
        return np.sum(self._tshape)*r


def canonical_error_data(T_can, T_full,rank_based=False,tol=1e-15,M=None,Norm="L2"):
    """
    @author : Lucas 27/06/18
    Builds a set of approximation error and associated compression rate for a
    representative subset of ranks.

    **Parameters**:
    *T_can*  Canonical approximation
    *T_full* Full tensor

    **Return** ndarray of error values, ndarray of compression rates

    **Todo** Add integration matrices to compute actual error associated with
    discrete integration operator
    """
    from pydecomp.core.tensor_algebra import norm
    data_compression=[]
    shape=T_full.shape
    mem_Full=np.product(shape)
    maxrank=T_can.get_rank()
    #skipping most ranks of higher values as they individually yield little information
    #this could be done more precisely with separated weight (related to scalar product)
    if rank_based:
        rank_list=np.arange(0,maxrank)
    else:
        rank_list=build_eval_rank_list(maxrank)

    error=[]
    T_norm=norm(T_full,M,type=Norm)
    comp_rate=[]
    for r in rank_list:
        if not rank_based:
            comp_rate.append(T_can.memory_eval(r)/mem_Full)

        T_approx=T_can.to_full_quick(r)
        actual_error=norm(T_full-T_approx,M,type=Norm)/T_norm
        error.append(actual_error)
        try:
            err_var=np.abs(error[-1]-error[-2])/error[-2]
            if comp_rate[-1]>1 or err_var<1e-10 or actual_error < tol:
                break
        except:
            pass

    if rank_based:
        return np.asarray(error)
    else:
        return np.asarray(error), np.asarray(comp_rate)

def build_eval_rank_list(maxrank):
    """
    @author : Lucas 27/06/18
    Sort of handwritten log spacing of rank evaluation.
    """
    if maxrank < 20:
        rank_list=[i for i in range(1,maxrank)]
    else:
        rank_list=[i for i in range(1,20)]
        try:
            for i in range(20,min(100, maxrank),5):
                rank_list.append(i)
        except:
            pass
        try:
            for i in range(100, maxrank,20):
                rank_list.append(i)
        except:
            pass
    return rank_list


def new_iteration_result(R, tshape, Resultat):
    """
    This functions serves to initiate le process of reconstruction in
    the reconstruction
    """
    NewModeResultat=(np.transpose(np.array([R[0]])).dot(np.array([R[1]])))

    if len(R)>2:
         for i in range(2,len(R)):
             NewModeResultat=np.kron(NewModeResultat,R[i])
         NewModeResultat=NewModeResultat.reshape(tshape)
    Resultat=np.add(Resultat,NewModeResultat)
    return Resultat


if __name__=="__main__":
    #@Diego, Write proper tests (creation, eval etc)
    print(build_eval_rank_list(300))

# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:25:40 2018

@author: Diego Britez
"""

from pydecomp.core.tensor_algebra import multilinear_multiplication
import pickle
import numpy as np
from pydecomp.utils.misc import rank_sampling

# @Diego Need for uniformization with decided structure for full format (ndarray)
class TuckerTensor():
    """
    This class is created to storage a decomposed Tensor in the Tucker
    Format, this is code is based in ttensor code from pytensor code package.\n
    **Attributes**\n
         **shape**: array like, with the numbers of elements that each 1-rank
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is
        going to be represented. The value of dim must be coherent with the
        size of _tshape parameter. \n

        **core:** List type, in this list will be storage the core of the
        decomposed tensor.\n

        **u:**List of the projection matrices in  each subspace.\n

    **Tucker Format Definition**\n
    """
    core = None;
    u = None;
#-----------------------------------------------------------------------------
    def __init__(self, core, uIn):
        #Handle if the uIn is not a list
        if(uIn.__class__ != list):
            uIn=[x for x in uIn]

        # check that each U is indeed a matrix
        for i in range(0,len(uIn)):
            if (len(uIn[i].shape) != 2):
                raise ValueError("{0} is not a 2-D matrix!".format(uIn[i]));

        # Size error checking
        k = core.shape;
        a="""Number of dims of Core and the number of matrices are different"""
        b="""{0} th dimension of Core is different from the number
            of columns of uIn[i]"""
        if (len(k) != len(uIn)):
            raise ValueError(a);

        for i in range(0,len(uIn)):
            if (k[i] != len((uIn[i])[0])):
                raise ValueError(b.format(i));

        self.ndim = core.ndim
        self.core = core.copy();
        self.rank = self.core.shape
        self.u = uIn;

        #save the shape of the ttensor
        shape = [];
        for i in range(0, len(self.u)):
            shape.extend([len(self.u[i])]);
        self.shape = tuple(shape);
        # constructor end #
#-----------------------------------------------------------------------------
    def size(self):
        ret = 1;
        for i in range(0, len(self.shape)):
            ret = ret * self.shape[i];
        return ret;
#-----------------------------------------------------------------------------
    def dimsize(self):
        return len(self.u)
#-----------------------------------------------------------------------------
    def copy(self):
        return TuckerTensor(self.core, self.u);
#-----------------------------------------------------------------------------
    def destructor(self):
        self.u=[]
        self.core=0
        self.shape=[]
#-----------------------------------------------------------------------------
    def reconstruction(self):
        """returns a FullFormat object that is represented by the
        tucker tensor"""
        dim=len(self.u)
        Fresult=multilinear_multiplication(self.u,self.core,dim)
        return Fresult

    def to_full(self):
        "Alias of reconstruction"
        return self.reconstruction()
#-----------------------------------------------------------------------------
    def __str__(self):
        ret = "Tucker tensor of size {0}\n".format(self.shape);
        ret += "Rank = {0} \n".format(self.rank);
        for i in range(0, len(self.u)):
            ret += "u[{0}] =\n{1}\n".format(i, self.u[i].shape);

        return ret;

    def memory_eval(self):
        "Returns the number of floats required to store self"
        mem=np.product(self.rank)
        for i in range(self.ndim):
            mem+=self.shape[i]*self.rank[i]
        return mem
    
    def save(self,path):
        with open(path,"wb") as f:
            pickle.dump(self,f)
        return

def tucker_error_data_complete(T_tucker, T_full,int_rules=None,sampling="exponential"):
    """ Computes the error data (error and compression rate) for Tucker
    decompositions

    **Parameters**
    T_tucker    [TuckerTensor] truncated tucker decomposition of T_full
    T_full      [ndarray]      original data
    int_rules   [MassMatrices] *optional*, if one wants to compute a norm diffrent
                from the frobenius norm, for discrete L2.

    **return**
    comp_rate   [list] Contains the compression rates for each error level
                (compressed_size/ full_size).
    error       [list] Compression error in 'F' norm or "int_rules" norm
    """
    # from numpy.linalg import norm
    from pydecomp.core.tensor_algebra import norm
    #We are going to calculate one average value of ranks
    d=T_full.ndim
    data_compression=[]
    shape=T_full.shape
    F_volume=np.product(shape)
    rank=np.asarray(T_tucker.rank)
    maxrank=max(rank)
    print("Computing approximation error chart of TT decomposition with {} sampling\n".format(sampling))
    print("With maxrank={}".format(maxrank))

    error=[]
    comp_rate=[]
    norm_full={"L1":norm(T_full,int_rules,type="L1"),
            "L2":norm(T_full,int_rules,type="L2"),
            "Linf":norm(T_full,int_rules,type="Linf")}
    r=np.zeros(d)
    actual_error={"L1":[],"L2":[],"Linf":[]}
    for i in rank_sampling(maxrank,sampling):
        r=np.minimum(rank,i)
        print(r)
        T_trunc=truncate(T_tucker,r)
        comp_rate.append(T_trunc.memory_eval()/F_volume)
        T_approx=T_trunc.reconstruction()
        actual_error["L1"].append(norm(T_full-T_approx,int_rules,type="L1")/norm_full["L1"])
        actual_error["L2"].append(norm(T_full-T_approx,int_rules,type="L2")/norm_full["L2"])
        actual_error["Linf"].append(norm(T_full-T_approx,int_rules,type="Linf")/norm_full["Linf"])
        del(T_approx)
    return actual_error, np.asarray(comp_rate)



def tucker_error_data(T_tucker, T_full, int_rules=None, Norm="L2",sampling="exponential"):
    """ Computes the error data (error and compression rate) for Tucker
    decompositions

    **Parameters**
    T_tucker    [TuckerTensor] truncated tucker decomposition of T_full
    T_full      [ndarray]      original data
    int_rules   [MassMatrices] *optional*, if one wants to compute a norm diffrent
                from the frobenius norm, for discrete L2.

    **return**
    comp_rate   [list] Contains the compression rates for each error level
                (compressed_size/ full_size).
    error       [list] Compression error in 'F' norm or "int_rules" norm
    """
    # from numpy.linalg import norm
    from pydecomp.core.tensor_algebra import norm
    #We are going to calculate one average value of ranks
    d=T_full.ndim
    data_compression=[]
    shape=T_full.shape
    F_volume=np.product(shape)
    rank=np.asarray(T_tucker.rank)
    maxrank=max(rank)

    error=[]
    comp_rate=[]

    norm_full=norm(T_full,int_rules,type=Norm)
    r=np.zeros(d)
    for i in rank_sampling(maxrank,sampling):
        r=np.minimum(rank,i)
        T_trunc=truncate(T_tucker,r)
        comp_rate.append(T_trunc.memory_eval()/F_volume)
        T_approx=T_trunc.reconstruction()
        actual_error=norm(T_full-T_approx,int_rules,type=Norm)/norm_full
        error.append(actual_error)
        del(T_approx)
    return np.asarray(error), np.asarray(comp_rate)


def truncate(T_tucker,trunc_rank):
    """Returns a truncated rank tucker tensor"""
    r=np.minimum(trunc_rank,T_tucker.rank)
    d=T_tucker.ndim
    core_shape=''
    modes=[]
    for j in range(d):
        modes.append(T_tucker.u[j][:,:int(r[j])])
        core_shape+=":"+str(r[j])
        if j<(d-1):
            core_shape+=','
    core=eval("T_tucker.core["+core_shape+"]")
    return TuckerTensor(core,modes)

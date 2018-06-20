#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/06/2018

@author: lucas
"""

import numpy as np
from tensor_descriptor_class import TensorDescriptor
from cls_RpodTree import  RpodTree
import numpy as np
import scipy
from POD import POD2 as POD
import high_order_decomposition_method_functions as hf
import copy


class recursive_tensor(TensorDescriptor):
    """
    ** Recursive Type Format**
    This format is designed to store and reconstruct RPOD decomposition.
    Essentially contains a nested list. Tries to follow as closely as possible
    the pure RPOD theory.

    **Attributes**

    **decomp** : RpodTree containing the decomposition. \n
    **sigma_max** : maximum sigma. \n
    **sigma_max_loc** : maximum sigma location. \n
    **rank** : nested list of ranks. \n
    **_tshape**: array like, with the numbers of elements that each 1-rank
    tensor is going to discretize each subspace of the full tensor. \n
    **dim**: integer type, number that represent the n-rank tensor that is
    going to be represented. The value of dim must be coherent with the
    size of _tshape parameter. \n
    """
    def __init__(self,_tshape,dim):
        TensorDescriptor.__init__(self,_tshape,dim)
        self.sigma_max=-1
        self.sigma_max_loc=[]
        self.rank=[]
        self.tree=None
        self.tree_weight=None

    def __str__(self):
        ret = "-----------------------------------------------\n"
        ret+= "Recursive_tensor of shape {0}\n\n".format(self._tshape);
        ret+= "Sigma_max={0}, sigma_max_loc={1}\n".format(self.sigma_max,self.sigma_max_loc)
        ret+= "Total tree weight (norm)={0} \n".format(self.tree_weight)
        ret+= "-----------------------------------------------\n"
        ret+= "Printing tree structure\n"
        ret+= "-----------------------------------------------\n"
        ret+= "node kind (branch_weight)\n"
        ret+= "-----------------------------------------------\n"
        ret+=str(self.tree)
        ret+= "-----------------------------------------------\n"
        return ret

    def to_full(self,rank=[]):
        return eval_rpod(self.tree, self._tshape)


def rpod(f, int_weights=None, POD_tol=1e-10, cutoff_tol=1e-10):
    """
    Performs the RPOD algorithm on f returning a recursive tensor

    **Attributes**

    **f**: ndarray input tensor \n
    **int_weights**: list of integratino weights \n
    **tol**: float, POD tolerance \n

    **Return ** recursive tensor containing the decomposition tree.
    """
    if int_weights==None:
        X=[np.ones(x) for x in f.shape]
        int_weights=[scipy.sparse.diags(x) for x in X]

    rpod_approx=recursive_tensor(f.shape,f.ndim)
    rpod_approx.tree = RpodTree(np.zeros(0))
    node_index = []
    rpod_rec(f, rpod_approx,int_weights, node_index, POD_tol, cutoff_tol)

    return rpod_approx

def rpod_rec(f, rpod_approx, int_weights, node_index, POD_tol=1e-10, cutoff_tol=1e-10):
    """
    Recursive part of the RPOD algorithm. Actually corresponds to the
    mathematical definition (see PhD. Thesis)
    **Attributes**

    **f**: ndarray input tensor \n
    **rpod_approx**: recursive_tensor containing the approximation of f \n
    **int_weights**: list of integratino weights \n
    **node_index**: indicates position in the tree \n
    **tol**: float, POD tolerance \n
    """
    Mx,Mt = hf.matricize_mass_matrix(f.ndim,0,int_weights)
    Phi = np.reshape(f, [f.shape[0], -1])

    U, sigma, V = POD(Phi, Mx, Mt, POD_tol)
    rank = U.shape[1]
    sigma_vec=sigma.diagonal()
    if rpod_approx.tree_weight==None: #initial
        rpod_approx.tree_weight=sigma_vec.sum()
    # sending weight to the last mode and keeping it as sigma
    if len(f.shape[1:]) == 1:
        if rpod_approx.sigma_max < sigma_vec[0]:
            rpod_approx.sigma_max=sigma_vec[0]
            rpod_approx.sigma_max_loc=np.concatenate([node_index,[0]])
    else:
        V = V@sigma
    Phi_shape = np.append(f.shape[1:], rank)
    Phi_next = [np.reshape(V[:, i], f.shape[1:]) for i in range(rank)]
    # recursive call on each phi_i
    for i in range(rank):
        branch_weight=sigma_vec[i]/rpod_approx.tree_weight
        #special case for root #making sure that a branch has at least one leaf
        if branch_weight < cutoff_tol:
            if node_index==[] or rpod_approx.tree.has_leaf(node_index):
                continue
        if len(f.shape[1:]) == 1:
            rpod_approx.tree.add_leaf(U[:, i], Phi_next[i], sigma_vec[i], node_index, branch_weight)
        else:
            rpod_approx.tree.add_node(U[:, i], node_index, branch_weight)
            node_index.append(i)
            rpod_rec(Phi_next[i], rpod_approx, int_weights[1:], node_index, POD_tol, cutoff_tol)
            node_index.pop()

def eval_rpod(tree: RpodTree, shape):
    res = eval_rpod_rec(tree)
    return np.reshape(res, shape)

def eval_rpod_rec(tree: RpodTree):
    if tree.is_last:
        child = tree.children[0]
        res = child.eval()
        for child in tree.children[1:]:
            res = res + child.eval()
        return res
    else:
        child = tree.children[0]
        res = np.kron(child.u, eval_rpod_rec(child))
        for child in tree.children[1:]:
            res = res + np.kron(child.u, eval_rpod_rec(child))
        return res


if __name__=='__main__':
    from benchmark_multivariable import benchmark_multivariable
    benchmark_multivariable(["RPOD"], ['trapezes'],dim=5 , shape=[115,35,21,7,5],
                                  test_function=2, plot="no",
                                  output_variable_file='yes',
                                  output_variable_name='test_rpod')

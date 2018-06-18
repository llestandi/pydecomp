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


class recursive_tensor(TensorDescriptor):
    """
    ** Recursive Type Format**
    This format is designed to store and reconstruct RPOD decomposition.
    Essentially contains a nested list. Tries to follow as closely as possible
    the pure RPOD theory.

    **Attributes**

    **decomp** : nested list of eigen functions. \n
    **weights** : nested list of singular values. \n
    **rpod_rank** : nested list of ranks. \n
    **_tshape**: array like, with the numbers of elements that each 1-rank
    tensor is going to discretize each subspace of the full tensor. \n
    **dim**: integer type, number that represent the n-rank tensor that is
    going to be represented. The value of dim must be coherent with the
    size of _tshape parameter. \n
    """
    def __init__(self,_tshape,dim):
        TensorDescriptor.__init__(self,_tshape,dim)
        self.rank=[]
        self.tree=None

    def __str__(self):
        ret = "ttensor of size {0}\n".format(self._tshape);
        ret+=str(self.tree)
        return ret

    def to_full(self,rank=[]):
        return eval_rpod(self.tree, self._tshape)


def rpod(f, int_weights, tol=1.e-3):
    """
    Performs the RPOD algorithm on f returning a recursive tensor

    **Attributes**

    **f**: ndarray input tensor \n
    **int_weights**: list of integratino weights \n
    **tol**: float, POD tolerance \n

    **Return ** recursive tensor containing the decomposition tree.
    """
    tree = RpodTree(np.zeros(0))
    node_index = []
    rpod_rec(f, tree,int_weights, node_index, tol)

    rpod_approx=recursive_tensor(f.shape,f.ndim)
    rpod_approx.tree=tree
    return rpod_approx

def rpod_rec(f, tree, int_weights, node_index, tol):
    f_shape = f.shape
    Mx,Mt = hf.matricize_mass_matrix(f.ndim,0,int_weights)
    Phi = np.reshape(f, [f.shape[0], -1])
    U, sigma, V = POD(Phi, Mx, Mt, tol)
    rank = U.shape[1]
    V = V@sigma
    Phi_shape = np.append(f_shape[1:], rank)
    Phi_next = [np.reshape(V[:, i], f_shape[1:]) for i in range(rank)]

    for i in range(rank):
        if len(f_shape[1:]) == 1:
            tree.add_leaf(U[:, i], Phi_next[i], node_index)
        else:
            tree.add_node(U[:, i], node_index)
            node_index.append(i)
            rpod_rec(Phi_next[i], tree, int_weights[1:], node_index, tol)
            node_index.pop()

def eval_rpod(tree: RpodTree, shape):
    res = eval_rpod_rec(tree)
    return np.reshape(res, shape)

def eval_rpod_rec(tree: RpodTree):
    if tree.is_last:
        child = tree.children[0]
        res = np.kron(child.u, child.v)
        for child in tree.children[1:]:
            res = res + np.kron(child.u, child.v)
        return res
    else:
        child = tree.children[0]
        res = np.kron(child.u, eval_rpod_rec(child))
        for child in tree.children[1:]:
            res = res + np.kron(child.u, eval_rpod_rec(child))
        return res


if __name__=='__main__':
    from benchmark_multivariable import benchmark_multivariable
    benchmark_multivariable(["RPOD"], ['trapezes'],dim=4 , shape=[115,10,8,20],
                                  test_function=1, plot="no",
                                  output_variable_file='yes',
                                  output_variable_name='test_rpod')

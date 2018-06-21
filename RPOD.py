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
        self.rank=[[] for i in range(dim-1)]
        self.tree=None
        self.tree_weight=None

    def __str__(self):
        ret = "-----------------------------------------------\n"
        ret+= "Recursive_tensor of shape {0}\n\n".format(self._tshape);
        ret+= "Sigma_max={0}, sigma_max_loc={1}\n".format(self.sigma_max,self.sigma_max_loc)
        ret+= "Total tree weight (norm)={0} \n".format(self.tree_weight)
        ret+= "Rank:\n {0}\n".format(self.rank)
        ret+= "-----------------------------------------------\n"
        ret+= "Printing tree structure\n"
        ret+= "-----------------------------------------------\n"
        ret+= "node kind (branch_weight)\n"
        ret+= "-----------------------------------------------\n"
        ret+=str(self.tree)
        ret+= "-----------------------------------------------\n"
        return ret

    def to_full(self, cutoff_tol=1e-16):
        return eval_rpod_tree(self.tree, self._tshape, cutoff_tol)

    def size(self):
        """ returns the number of elements stored """
        shape=self._tshape
        buff=np.sum(np.asarray(shape[-1]*self.rank[-1]))
        for i in range(self._dim-1):
            print(i)
            buff+=np.sum(np.asarray(shape[i]*self.rank[i]))
        return buff

    def compression_rate(self):
        """ Compression rate as compared to the full tensor size"""
        return self.size()/np.product(self._tshape)


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
    ######## POD part ################
    Mx,Mt = hf.matricize_mass_matrix(f.ndim,0,int_weights)
    Phi = np.reshape(f, [f.shape[0], -1])

    U, sigma, V = POD(Phi, Mx, Mt, POD_tol)
    pod_rank = U.shape[1]

    ######## Preparing recursive call ########
    loc_rank =0
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
    Phi_shape = np.append(f.shape[1:], pod_rank)
    Phi_next = [np.reshape(V[:, i], f.shape[1:]) for i in range(pod_rank)]

    ######### recursive call on each phi_i ########
    for i in range(pod_rank):
        branch_weight=sigma_vec[i]/rpod_approx.tree_weight
        #special case for root and making sure that a branch has at least one leaf
        if branch_weight < cutoff_tol:
            if node_index==[] or rpod_approx.tree.has_leaf(node_index):
                continue
        #core action
        if len(f.shape[1:]) == 1:
            rpod_approx.tree.add_leaf(U[:, i], Phi_next[i], sigma_vec[i], node_index, branch_weight)
        else:
            rpod_approx.tree.add_node(U[:, i], node_index, branch_weight)
            node_index.append(i)
            rpod_rec(Phi_next[i], rpod_approx, int_weights[1:], node_index, POD_tol, cutoff_tol)
            node_index.pop()
        loc_rank+=1
    rpod_approx.rank[len(node_index)].append(loc_rank)


def eval_rpod_tree(tree: RpodTree, shape, cutoff_tol=1e-16):
    """ Reconstruct tensor to full format while ignoring branches
    with weight lower than cutoff_tol """
    res = eval_rpod_rec_tree(tree,cutoff_tol)
    return np.reshape(res, shape)

def eval_rpod_rec_tree(tree: RpodTree, cutoff_tol=1e-16):
    if tree.is_last:
        child = tree.children[0]
        res = child.eval()
        for child in tree.children[1:]:
            if child.branch_weight>cutoff_tol:
                res = res + child.eval()
        return res
    else:
        child = tree.children[0]
        res = np.kron(child.u, eval_rpod_rec_tree(child,cutoff_tol))
        for child in tree.children[1:]:
            if child.branch_weight>cutoff_tol:
                res = res + np.kron(child.u, eval_rpod_rec_tree(child,cutoff_tol))
        return res

        return np.reshape(res, rec_tens._tshape)



if __name__=='__main__':
    from benchmark_multivariable import benchmark_multivariable
    benchmark_multivariable(["RPOD"], ['trapezes'],dim=3 , shape=[100,100,100],
                                  test_function=1, plot="no",
                                  output_variable_file='yes',
                                  output_variable_name='test_rpod')

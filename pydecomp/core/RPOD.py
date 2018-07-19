#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/06/2018

@author: lucas
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import core.MassMatrices as mm
from deprecated.tensor_descriptor_class import TensorDescriptor
from core.cls_RpodTree import  RpodTree
from core.POD import POD
from core.MassMatrices import pop_1_MM

class RecursiveTensor(TensorDescriptor):
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
    **shape**: array like, with the numbers of elements that each 1-rank
    tensor is going to discretize each subspace of the full tensor. \n
    **dim**: integer type, number that represent the n-rank tensor that is
    going to be represented. The value of dim must be coherent with the
    size of shape parameter. \n
    """
    def __init__(self,shape):
        self.shape=shape
        self.ndim=len(shape)
        self.sigma_max=-1
        self.sigma_max_loc=[]
        self.rank=[[] for i in range(self.ndim-1)]
        self.tree=None
        self.tree_weight=None

    def __str__(self):
        ret = "-----------------------------------------------\n"
        ret+= "Recursive_tensor of shape {0}\n\n".format(self.shape);
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
        return eval_rpod_tree(self.tree, self.shape, cutoff_tol)

    def size(self):
        """ returns the number of elements stored """
        shape=self.shape
        buff=np.sum(np.asarray(shape[-1]*self.rank[-1]))
        for i in range(self._dim-1):
            buff+=np.sum(np.asarray(shape[i]*self.rank[i]))
        return buff

    def compression_rate(self):
        """ Compression rate as compared to the full tensor size"""
        return self.size()/np.product(self.shape)


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

    rpod_approx=RecursiveTensor(f.shape)
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
    Mx,Mt = mm.matricize_mass_matrix(f.ndim,0,int_weights)
    Phi = np.reshape(f, [f.shape[0], -1])

    U, sigma, V = POD(Phi, Mx, Mt, POD_tol)
    pod_rank = U.shape[1]

    ######## Preparing recursive call ########
    try :
        sigma_vec=sigma.diagonal()
    except:
        sigma_vec=sigma.copy()
    if rpod_approx.tree_weight==None: #initial
        rpod_approx.tree_weight=sigma_vec.sum()

    at_leaf=(len(f.shape[1:]) == 1)
    eval_sigma_max(rpod_approx,sigma_vec,at_leaf,node_index)
    if not at_leaf:
        V = V*sigma

    Phi_shape = np.append(f.shape[1:], pod_rank)
    Phi_next = [np.reshape(V[:, i], f.shape[1:]) for i in range(pod_rank)]

    ######### recursive call on each phi_i ########
    loc_rank =0
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
            rpod_rec(Phi_next[i], rpod_approx, pop_1_MM(int_weights), node_index, POD_tol, cutoff_tol)
            node_index.pop()
        loc_rank+=1
    rpod_approx.rank[len(node_index)].append(loc_rank)

def eval_sigma_max(rpod_approx,sigma_vec, at_leaf,node_index):
    """ Test and updates branches weights in rpod algorithm """

    # sending weight to the last mode and keeping it as sigma
    if at_leaf:
        if rpod_approx.sigma_max < sigma_vec[0]:
            rpod_approx.sigma_max=sigma_vec[0]
            rpod_approx.sigma_max_loc=np.concatenate([node_index,[0]])

def eval_rpod_tree(tree: RpodTree, shape, cutoff_tol=1e-16):
    """ Reconstruct tensor to full format while ignoring branches
    with weight lower than cutoff_tol """
    res = eval_rpod_rec_tree(tree,cutoff_tol)
    return np.reshape(res, shape)

def eval_rpod_rec_tree(tree: RpodTree, cutoff_tol=1e-16):
    if tree.is_last:
        #naive and presumably slow version
        #makes sure that an explored branch returns something
        child = tree.children[0]
        res = child.eval()
        for child in tree.children[1:]:
            if child.branch_weight>cutoff_tol:
                res = res + child.eval()
        return res
        # THis one is left here as an example to improve efficiecy. not working, the
        # leaves themselves should be changed
        # firstborn=tree.children[0]
        # U = np.expand_dims(firstborn.u,axis=1)
        # V = np.expand_dims(firstborn.sigma*firstborn.v,axis=1)
        # for child in tree.children[1:]:
        #     if child.branch_weight>cutoff_tol:
        #         U=np.concatenate([U,np.expand_dims(child.u,axis=1)],axis=1)
        #         V=np.concatenate([V,np.expand_dims(child.sigma*child.v,axis=1)],axis=1)
        # return np.matmul(U,V.T)
    else:
        child = tree.children[0]
        res = np.kron(child.u, eval_rpod_rec_tree(child,cutoff_tol))
        for child in tree.children[1:]:
            if child.branch_weight>cutoff_tol:
                res = res + np.kron(child.u, eval_rpod_rec_tree(child,cutoff_tol))
        return res

def rpod_tree_size(tree: RpodTree, shape,cutoff_tol=1e-8):
    """ Evaluate RpodTree size while ignoring branches
    with weight lower than cutoff_tol """
    return rpod_size_rec(tree,cutoff_tol)/np.product(shape)


def rpod_size_rec(tree: RpodTree, cutoff_tol=1e-8):
    size=0
    for child in tree.children:
        if child.branch_weight>cutoff_tol:
            if tree.is_last:
                size+= child.u.size+child.v.size
            else:
                size += child.u.size+ rpod_size_rec(child,cutoff_tol)
    return size

def rpod_error_data(T_rec,T_full,min_tol=1.,max_tol=1e-8):
    """Computes error and compression rate with given tolerance"""
    err=[]
    comp_rate=[]
    norm_T=np.linalg.norm(T_full)
    tol_space=np.logspace(np.log10(min_tol), np.log10(max_tol))

    for tol in tol_space:
        loc_rate=rpod_tree_size(T_rec.tree,T_rec.shape,tol)
        if loc_rate==0:
            continue #prevent cutoff discrepencies
        elif not comp_rate:
            pass
        elif loc_rate==comp_rate[-1]:
            continue

        err.append(np.linalg.norm(T_rec.to_full(tol)-T_full)/norm_T)
        comp_rate.append(loc_rate)
    return np.asarray(err), np.asarray(comp_rate)

def plot_rpod_approx(T_rec, T_full, out_file='RPOD_approx_error', min_tol=1., max_tol=1e-8):
    err,comp_rate=rpod_error_data(T_rec,T_full, min_tol,max_tol)
    print(err)
    print(comp_rate)
    color_code='m'
    marker_code='h'
    linestyle_code='--'
    label_line='RPOD'
    if plt.fignum_exists(1):
        if comp_rate[-1]>plt.axis()[1]:
            plt.xlim(0,(comp_rate[-1]+5))
    else:
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.set_xlim(0,(comp_rate[-1]+5))
        ax.set_ylim([1e-8,0.15])


    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()
    plt.legend()
    plt.plot(100*comp_rate, err ,color=color_code, marker=marker_code,
             linestyle=linestyle_code,
             label=label_line)
    plt.show()

    #saving plot as pdf
    plt.savefig(out_file)
    plt.close()
    return



if __name__=='__main__':
    from benchmark_multivariable import benchmark_multivariable
    benchmark_multivariable(["RPOD"], ['trapezes'],shape=[20,20,20],
                                  test_function=2, plot=False,output_decomp='',
                                  plot_name='output/approx_test_rpod.pdf')

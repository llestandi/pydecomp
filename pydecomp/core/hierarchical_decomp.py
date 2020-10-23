# -*- coding:utf-8 -*-
# @author: Lucas
# Adapted the HT tucker decomposition from YIN MIAO "https://github.com/ymthink/htpy"
# Lucas: 2020/10/22


import numpy as np
from math import log
from core.TSVD import TSVD
import core.tensor_algebra as ta
from core.tucker_decomp import THOSVD
from core.Tucker import TuckerTensor
import scipy.io as sio
from scipy.linalg import svd
from collections import deque
from utils.bytes2human import bytes2human


class HierarchicalTensor():
    """
    Wrapper for hierarchical tensor, building on node class. Very similar to
    RecursiveTensor but uses binary tree and follows Kressner and Tobler
    architecture for evenly distributed tensors. Far less complete that their
    implementation `htucker` , the only .

    Implicitly a binary tree
    """
    def __init__(self,shape):
        self.shape=shape
        self.ndim=len(shape)
        self.rank=[]
        self.depth=int(np.ceil(log(self.ndim,2)))
        self.root=None
        self.memory_use=0
        self.Tucker_memory_use=0

    def set_tree(self,root):
        self.root=root
        self.set_rank()

    def set_rank(self):
        for leaf in Node.find_leaf(self.root):
            self.rank.append(leaf.rank)

    def memory_eval(self):
        mem=0
        mem_basis=0
        for level in range(self.depth-1):
            for node in Node.find_cluster(self.root, level):
                mem+=node.b.size
            for leaf in Node.find_leaf(self.root):
                mem+=leaf.u.size
                mem_basis+=leaf.u.size
        self.memory_use=mem
        self.Tucker_memory_use=np.product(self.rank)+mem_basis

    def __str__(self):
        rep = "-----------------------------------------------\n"
        rep+= "HierarchicalTensor of shape {0}\n\n".format(self.shape);
        rep+= "Rank:\n {0}\n".format(self.rank)
        rep+= "Depth= {}\n\n".format(self.depth)
        rep+= "Memory use (FullFormat):{} ({})\n".format(bytes2human(self.memory_use * 8), bytes2human(np.product(self.shape)*8))
        rep+= "Compression rate (CR) :{:.2f}% \n".format(self.memory_use/np.product(self.shape)*100)
        rep+= "{:.2f} times more efficient than tucker Tucker \n".format(self.Tucker_memory_use/self.memory_use)
        rep+= "-----------------------------------------------\n"
        rep+= "Printing tree structure, strating from root\n"
        rep+= "-----------------------------------------------\n"
        for level in range(self.depth-1):
            for node in Node.find_cluster(self.root, level):
                rep+=str(node)
            rep+= "-----------------------------------------------\n"
            rep+="Increasing depth to {}\n".format(level+1)
        rep+= "Printing leaves\n"
        for leaf in Node.find_leaf(self.root):
            rep+=str(leaf)
        return rep


    def to_full(self):
        """ leaf to root reconstruction HT tensor.
        for each node, building U_t space from children U_l, U_r AND cluster tensor B
        **return** [ndarray], full order representation of HT"""
        for level in range(self.depth, -1, -1):
            for node in Node.find_cluster(self.root, level):
                #for each node, building U_t space from children U_l, U_r AND cluster tensor B
                shape_b = np.array(np.shape(node.b))
                b_mat = np.reshape(node.b, [node.left.rank, np.prod(shape_b[1:])])

                shape_left = np.array(np.shape(node.left.u))
                left_mat = np.reshape(node.left.u, [np.prod(shape_left[:-1]), node.left.rank])

                left_b_mat = np.matmul(left_mat, b_mat)
                left_b_mat = np.reshape(left_b_mat, [np.prod(shape_left[:-1]), node.right.rank, node.rank])
                left_b_mat = np.transpose(left_b_mat, [0, 2, 1])
                left_b_mat = np.reshape(left_b_mat, [np.prod(shape_left[:-1])*node.rank, node.right.rank])

                shape_right = np.array(np.shape(node.right.u))
                right_mat = np.reshape(node.right.u, [np.prod(shape_right[:-1]), node.right.rank])
                right_mat = np.transpose(right_mat)

                left_b_right_mat = np.matmul(left_b_mat, right_mat)
                left_b_right_mat = np.reshape(left_b_right_mat, [np.prod(shape_left[:-1]), node.rank, np.prod(shape_right[:-1])])

                new_shape = np.concatenate([shape_left[:-1], [node.rank], shape_right[:-1]])
                left_b_right = np.reshape(left_b_right_mat, new_shape)
                rank_index = len(shape_left[:-1])
                new_indices = np.concatenate([np.arange(0, rank_index), np.arange(rank_index+1, len(new_shape)), [rank_index]])
                left_b_right = np.transpose(left_b_right, new_indices)

                node.set_u(left_b_right) # for internal nodes, u is a representation of the whole U_t space, with t={set of dimension}

        x_ht = np.squeeze(self.root.u)
        return x_ht


class Node:
    def __init__(self, left, right, indices, rank, is_leaf, level):
        self.left = left
        self.right = right
        self.indices = indices
        self.rank = rank
        self.is_leaf = is_leaf
        self.level = level
        self.u = None
        self.b = None

    def set_rank(self, rank):
        self.rank = rank

    def set_u(self, u):
        self.u = u

    def set_b(self, b):
        self.b = b

    def __str__(self):
        rep ="Node {}\n".format(self.indices)
        rep+="is leaf :{}\n".format(self.is_leaf)
        rep+="rank:{}\n".format(self.rank)
        if self.is_leaf:
            rep+="With U.shape={}\n".format(self.u.shape)
        else:
            rep+="with B.shape={}\n".format(self.b.shape)
        return rep

    @staticmethod
    def indices_tree(n_mode):
        s = []
        root = Node(None, None, np.arange(0, n_mode), 0, False, 0)
        s.append(root)
        level_max = 0
        while len(s) > 0:
            cur_node = s.pop()
            if cur_node.level > level_max:
                level_max = cur_node.level
            if len(cur_node.indices) > 1:
                mid = len(cur_node.indices) // 2
                left_node = Node(None, None, np.arange(cur_node.indices[0], cur_node.indices[mid]), 0, False, cur_node.level+1)
                right_node = Node(None, None, np.arange(cur_node.indices[mid], cur_node.indices[-1]+1), 0, False, cur_node.level+1)
                cur_node.left = left_node
                cur_node.right = right_node
                s.append(left_node)
                s.append(right_node)
            else:
                cur_node.is_leaf = True

        return root, level_max

    @staticmethod
    def find_leaf(root):
        q = deque()
        q.append(root)
        while len(q) > 0:
            cur_node = q.popleft()
            if cur_node.is_leaf:
                yield cur_node
            else:
                q.append(cur_node.left)
                q.append(cur_node.right)

    @staticmethod
    def find_cluster(root, level):
        q = deque()
        q.append(root)
        while len(q) > 0:
            cur_node = q.popleft()
            if not cur_node.is_leaf:
                q.append(cur_node.left)
                q.append(cur_node.right)
                if cur_node.level == level:
                    yield cur_node


# leaf to root truncation
def compute_HT_decomp(x, epsilon=1e-4, eps_tuck=None, rmax=100, solver='EVD'):

    if type(x)==np.ndarray:
        #x_ = np.copy(x)
        if eps_tuck==None: eps_tuck=epsilon
        ##compute leaf decomposition by HOSVD first
        tucker= THOSVD(x,eps_tuck, rank=100, solver='EVD')
        print("tucker decomposition CR={:.2f}%".format(100*tucker.memory_eval()/np.product(x.shape)))
        print("Tucker error:{:.2e}".format(np.linalg.norm(tucker.to_full()-x)/np.linalg.norm(x)))
    elif type(x)==TuckerTensor:
        tucker= np.copy(x)

    shape = np.array(x.shape)
    n_mode = len(shape)
    root, level_max = Node.indices_tree(n_mode)
    x=x_=tucker.core
    for node in Node.find_leaf(root):
        dim=node.indices[0]
        node.rank=tucker.rank[dim]
        node.set_u(tucker.u[dim])

    rmax -= 1
    # compute cluster decomposition
    for level in range(level_max, -1, -1):
        count = 0
        for node in Node.find_cluster(root, level):
            if level < (level_max - 1):
                cur_indices = np.arange(2*count, 2*(count+1))
                cur_mode = 2 ** (level + 1)
            else:
                cur_indices = node.indices
                cur_mode = n_mode

            #rehsaping xmat for particular index, bit more complez than a matricization
            shape_core = np.array(np.shape(x))
            other_indices = np.concatenate([np.arange(0, cur_indices[0]), np.arange(cur_indices[-1]+1, cur_mode)])
            trans_indices = np.concatenate([cur_indices, other_indices])
            x_mat = np.transpose(x, trans_indices)
            x_mat = np.reshape(x_mat, [np.prod(shape_core[cur_indices]), np.prod(shape_core[other_indices])])

            if level == 0:
                node.rank = 1
                u = x_mat
            else:
                u, s, v = TSVD( x_mat, epsilon=epsilon, rank=rmax, solver=solver)
                vt = v.T
                node.rank = s.size

            #setting b
            b = np.reshape(u, [node.left.rank, node.right.rank, node.rank])
            node.set_b(b)

            #for each cluster, compute the part of the "projection" Cour certainly be cleaned but not useful for me
            shape_core = np.array(np.shape(x_))
            cur_mode -= count
            cur_indices -= count
            other_indices = np.concatenate([np.arange(0, cur_indices[0]), np.arange(cur_indices[-1]+1, cur_mode)])
            trans_indices = np.concatenate([cur_indices, other_indices])

            x_mat_ = np.transpose(x_, trans_indices)
            x_mat_ = np.reshape(x_mat_, [np.prod(shape_core[cur_indices]), np.prod(shape_core[other_indices])])
            u_x_mat_ = np.matmul(u.T, x_mat_)

            new_indices = []
            for i in range(len(trans_indices)):
                if trans_indices[i] < cur_indices[0]:
                    new_indices.append(trans_indices[i])
                elif trans_indices[i] > cur_indices[-1]:
                    new_indices.append(trans_indices[i]-len(cur_indices)+1)
                elif trans_indices[i] == cur_indices[0]:
                    new_indices.append(cur_indices[0])

            trans_indices = []
            for i in range(len(new_indices)):
                for j in range(len(new_indices)):
                    if new_indices[j] == i:
                        trans_indices.append(j)
                        break

            new_shape = np.concatenate([[node.rank], shape_core[other_indices]])
            x_ = np.reshape(u_x_mat_, new_shape)
            x_ = np.transpose(x_, trans_indices)
            count += 1

        x = x_

    HT=HierarchicalTensor(shape)
    HT.set_tree(root)
    HT.memory_eval()
    return HT

def build_error_data(x,eps_list=[1e-2,1e-4,1e-8],eps_tuck=1e-4,rmax=200):
    """
    Computes approximation for epsilon consequence and return error data.
    Since trunction is not straightforward, HOSVD is kept and a new HT L2R
    truncation is computed for each epsilon.

    **inputs**
        - x : ndarray containing data of intest
        - eps_list : list of epsilon to be sampled
        - eps_tuck : epsilon used in the unique HOSVD
        - rmax : is the maximum rank for any node, particularly restrictive for middle nodes

    **return** dict of relative error for L1,L2,Linf and associated compression rate
    """

if __name__ == '__main__':
    # x = sio.loadmat('x.mat')['x']
    n=50
    x = np.random.random([n, n, n, n])
    x = np.random.random([30, 5 ,40, 9, 10])
    #sio.savemat('x.mat', {'x':x})
    HT = compute_HT_decomp(x, epsilon=1e-6,eps_tuck=1e-3, rmax=200)
    print(HT)
    x_ht = HT.to_full()

    err = np.linalg.norm(x_ht - x)#/np.linalg.norm(x)
    print(err)
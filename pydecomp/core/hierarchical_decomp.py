# -*- coding:utf-8 -*-
# @author: Lucas
# Adapted the HT tucker decomposition from YIN MIAO "https://github.com/ymthink/htpy"
# Lucas: 2020/10/22


import numpy as np
from math import log
from core.TSVD import TSVD
from core.tensor_algebra import norm
from core.tucker_decomp import THOSVD
from core.Tucker import TuckerTensor, truncate
import scipy.io as sio
from scipy.linalg import svd
from collections import deque
from utils.bytes2human import bytes2human
from copy import deepcopy
from analysis.plot import rank_benchmark_plotter
import time


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
        self.compression_rate=mem/np.product(self.shape)

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
        for level in range(self.depth+1):
            rep+="Level {}\n".format(level+1)
            for node in Node.find_cluster(self.root, level):
                rep+=str(node)
            for leaf in Node.find_leaf(self.root):
                if leaf.level == level:
                    rep+=str(leaf)
            rep+= "-----------------------------------------------\n"
        # for leaf in Node.find_leaf(self.root):
            # rep+=str(leaf)
        return rep

    def get_rank(self):
        "return a dictionnary of nodes with their rank"
        rank={}
        for level in range(1,self.depth+1):
            for node in Node.find_cluster(self.root, level):
                try:
                    rank[str(node.indices)]=node.rank
                except:
                    rank[str(node.indices)]=node.s.size
        for node in Node.find_leaf(self.root):
            rank[str(node.indices)]=node.rank
        return rank

    def plot_singular_values(self,show=True,type="interpretation",plot_name="figures/HT_sing_vals"):
        if type=="level":
            #pure level loop
            for level in range(1,self.depth+1):
                rank_data={}
                for node in Node.find_cluster(self.root, level):
                    index=str(node.indices)
                    rank_data[index]=node.s
                for leaf in Node.find_leaf(self.root):
                    if leaf.level == level:
                        index=str(leaf.indices)
                        rank_data[index]=leaf.s
                rank_benchmark_plotter(rank_data, show=True,
                                    plot_name=plot_name+"level_{}.pdf".format(level),
                                    title="Level {} nodes and leaves".format(level),
                                    ylabel="Singular Values")

        if type=="interpretation":
            #looping on clusters per level
            for level in range(1,self.depth):
                rank_data={}
                for node in Node.find_cluster(self.root, level):
                    index=str(node.indices)
                    rank_data[index]=node.s
                try:
                    rank_benchmark_plotter(rank_data, show=True,
                                        plot_name=plot_name+"level_{}.pdf".format(level),
                                        title="level {} nodes".format(level),
                                        ylabel="Singular Values")
                except:
                    print("Nothing to plot, skipping "+index)

            rank_data={}
            for leaf in Node.find_leaf(self.root):
                index=str(leaf.indices)
                rank_data[index]=leaf.s
                plot_name+="leaves.pdf"
            rank_benchmark_plotter(rank_data, show=True,
                        plot_name=plot_name+"leaf_plot.pdf".format(level),
                        title="Leaf plot nodes".format(level),
                        ylabel="Singular Values")

        return


    def to_full(self,clean=True):
        """ leaf to root reconstruction HT tensor.
        for each node, building U_t space from children U_l, U_r AND cluster tensor B
        **return** [ndarray], full order representation of HT"""
        
        if self.root.u is None:
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
        #cleaning up so as not to clog the memory
        if clean:
            for level in range(self.depth, -1, -1):
                for node in Node.find_cluster(self.root, level):
                    node.set_u(None)
            
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
        self.s=None
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
        try:
            rep+="Sigma[0:8]:{}\n".format(self.s[0:8])
        except:
            rep+="No Sigma\n"
        if self.is_leaf:
            rep+="With U.shape={}\n".format(self.u.shape)
        else:
            try: 
                rep+="With U.shape={}\n".format(self.u.shape)
            except :
                pass
            rep+="with B.shape={}\n".format(self.b.shape)
            if self.level==0:
                rep+=str(self.b[:,:,0])+"\n"
        return rep

    def plot_singular_values(self,show=True,plot_name=""):
        index="node_"+str(self.indices)
        rank_data={index:self.s}
        print(index)
        plot_name+=index+".pdf"
        try:
            rank_benchmark_plotter(rank_data, show=True, plot_name=plot_name)
        except:
            print("Nothing to plot, skipping "+index)

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
def compute_HT_decomp(x, epsilon=1e-4, eps_tuck=None, rmax=100, solver='EVD',verbose=0):
    if type(x)==np.ndarray:
        #x_ = np.copy(x)
        if eps_tuck==None: eps_tuck=epsilon
        ##compute leaf decomposition by HOSVD first
        tucker,sigma= THOSVD(x,eps_tuck, rank=rmax, solver='EVD',export_s=True)
        if verbose>0:
            print("tucker decomposition CR={:.2f}%".format(100*tucker.memory_eval()/np.product(x.shape)))
            print("Tucker error:{:.2e}".format(np.linalg.norm(tucker.to_full()-x)/np.linalg.norm(x)))
    elif type(x)==TuckerTensor:
        tucker= deepcopy(x)
    elif type(x)==list or type(x)==tuple:
        tucker= deepcopy(x[0])
        sigma= x[1]
    else:
        print(type(x))
        raise Exception("something went wrong with x type")

    shape = np.array(tucker.shape)
    n_mode = len(shape)
    root, level_max = Node.indices_tree(n_mode)
    x_=tucker.core
    x=x_
    for node in Node.find_leaf(root):
        dim=node.indices[0]
        node.rank=tucker.rank[dim]
        node.set_u(tucker.u[dim])
        try:
            node.s=sigma[dim]
        except:
            node.s=None

    rmax -= 1
    # compute cluster decomposition
    for level in range(level_max, -1, -1):
        count = 0
        for node in Node.find_cluster(root, level):
            start_time=time.time()
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
                node.s=1
            else:
                u, s, v = TSVD( x_mat, epsilon=epsilon, rank=rmax, solver='EVD')
                if verbose>1:
                    print(np.log10(s))
                vt = v.T
                node.rank = s.size
                node.s=s

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
            x_mat_ = np.reshape(x_mat_, [np.prod(shape_core[cur_indices]), np.prod(shape_core[other_indices])]) #this one needs not be copied (good news since it may be large)
            uT=np.copy(u.T) #significantly speedup next matmul
            u_x_mat_ = uT @ x_mat_

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
            if verbose>0:
                print("Node {} has been computed in {}".format(node.indices, time.time()-start_time))

        x = x_

    HT=HierarchicalTensor(shape)
    HT.set_tree(root)
    HT.memory_eval()
    return HT

def HT_build_error_data(x,eps_list=[1e-2,1e-4,1e-8],mode="heterogenous",eps_tuck=1e-4,rmax=200,verbose=0):
    """
    Computes approximation for epsilon consequence and return error data.
    Since trunction is not straightforward, HOSVD is kept and a new HT L2R
    truncation is computed for each epsilon.

    **inputs**
        - x : ndarray containing data of intest
        - eps_list : list of epsilon to be sampled
        - mode :["heterogenous" for a single fine eps_tuck, "homogenous" for eps_tuck=eps]
        - eps_tuck : epsilon used in the unique HOSVD
        - rmax : is the maximum rank for any node, particularly restrictive for middle nodes

    **return** dict of relative error for L1,L2,Linf and associated compression rate
    """
    if type(x)==TuckerTensor:
        tucker=x
    else :
        print("computing Tucker decomposition with eps={}".format(eps_tuck))
        tucker,sigma=THOSVD(x,eps_tuck, rank=rmax, solver='EVD',export_s=True)
    print("Computed Common Tucker decomposition")
    print("Tucker rank :{}".format(tucker.rank))

    norm_full={"L1":norm(x,type="L1"),
            "L2":norm(x,type="L2"),
            "Linf":norm(x,type="Linf")}
    print("Tucker approx error is {:.2e}".format(norm(x-tucker.to_full(),type="L2"),norm_full["L2"]))
    print("Now computing HT "+mode)
    actual_error={"L1":[],"L2":[],"Linf":[]}
    comp_rate=[]
    for eps in eps_list:
        if mode=="homogenous":
            #cutting to epsilon
            sigma_work=[s/s[0] for s in sigma]
            sigma_work=[s[(s>eps)] for s in sigma_work]
            trunc_rank=[len(s) for s in sigma_work]
            tucker_work=truncate(tucker,trunc_rank)
        else:
            tucker_work=tucker
            sigma_work=sigma
        if verbose>0:
            print("-----------------------------------")
            print("Running HT for eps={}".format(eps))
            print("With leaf rank: ".format(tucker_work.rank))
        HT=compute_HT_decomp([tucker_work,sigma_work],eps,rmax=rmax)
        print(HT.get_rank())
        reconstruction=HT.to_full()
        comp_rate.append(HT.compression_rate)
        actual_error["L1"].append(norm(x-reconstruction,type="L1")/norm_full["L1"])
        actual_error["L2"].append(norm(x-reconstruction,type="L2")/norm_full["L2"])
        actual_error["Linf"].append(norm(x-reconstruction,type="Linf")/norm_full["Linf"])
        if verbose>2:
            print(HT.rank)

    return actual_error, np.asarray(comp_rate), HT


if __name__ == '__main__':
    # x = sio.loadmat('x.mat')['x']
    n=50
    x = np.random.random([n, n, n, n])
    x = np.random.random([30, 5 ,40, 9])
    #sio.savemat('x.mat', {'x':x})
    results = build_error_data(x,eps_list=[1e-2,1e-4,1e-8],eps_tuck=1e-16,rmax=200)
    print(results[:1])
    #
    # HT = compute_HT_decomp(x, epsilon=1e-6,eps_tuck=1e-3, rmax=200)
    # print(HT)
    # x_ht = HT.to_full()
    #
    # err = np.linalg.norm(x_ht - x)#/np.linalg.norm(x)
    # print(err)

import numpy as np

class RpodTree:
    """
    RPOD decomposition tree storage structure.
    Root node is empty. As a consequence of the RPOD algorithm, each node
    is built as follow:\n
    from f(x,t)=sum_i phi_i(x) a_i(t)\n
    - u stores [a_k] for all k< r\n
    - children point to phi_k(x) decompostion\n
    - is_last is just a trick to tell a nodes children are leaves\n
    leaves are described in a separate structure.

    **TODO**: For efficiency reasons, it would be better to store a POD decomp
    directly at leaf instead of having r leaves per terminal node. Same storage cost.
    Probably more efficient for evaluation.
    """
    def __init__(self, u, branch_weight=None):
        self.u = u
        self.children = []
        self.is_last = False
        #Cosmetic Attribute
        self.branch_weight=branch_weight

    def add_node(self, u, index, branch_weight=None):
        if len(index) == 0:
            self.children.append(RpodTree(u,branch_weight))
            self.is_last = False
        else:
            self.children[index[0]].add_node( u, index[1:],branch_weight)

    def add_leaf(self, u, v,sigma, index, branch_weight=None):
        """
        Adds a leaf which contains
        u :(1d array) mode on dim d -1
        v :(1d array) mode on dim d
        sigma: leaf weight
        """
        if len(index) == 0:
            self.children.append(RpodLeaf(u, v, sigma, branch_weight))
            self.is_last = True
        else:
            self.children[index[0]].add_leaf(u, v, sigma, index[1:], branch_weight)

    def __str__(self, string=''):
        list_tree = [[]]
        depth = 0
        self._rec_str(list_tree, depth)
        string = ''
        for i in range(len(list_tree)):
            string+="Level {0} :\n".format(i+1)
            for j in range(len(list_tree[i])):
                string += ' '+list_tree[i][j]
            string += ""
        return string

    def _rec_str(self, list_tree, depth):
        if self.is_last:
            for i in range(len(self.children)):
                list_tree[depth].append("leaf[{0}] ({1:2.3e}) \t".format(i,self.children[i].branch_weight))
                # list_tree[depth].append("U: {0} \nV: {1}\n\n".format(self.children[i].u, self.children[i].v))
            list_tree[depth].append('|\n')
        else:
            for i in range(len(self.children)):
                list_tree[depth].append('node ({0:2.3e})\t'.format(self.children[i].branch_weight))
            list_tree[depth].append('|\n')
            if len(list_tree) == depth+1:
                list_tree.append([])
            for i in range(len(self.children)):
                self.children[i]._rec_str(list_tree, depth+1)


class RpodLeaf:
    """ Leaf class to store RPOD decomp.
    Please note that the modes have been normalized and that sigma is separated.
    u : mode on dim d -1
    v : mode on dim d
    sigma: leaf weight
    """
    def __init__(self, u, v, sigma=1,branch_weight=None):
        self.u = u
        self.v = v
        self.sigma= sigma
        self.branch_weight=branch_weight

    def eval(self):
        return self.sigma*(np.kron(self.u,self.v))

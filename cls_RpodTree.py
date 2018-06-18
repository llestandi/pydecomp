class RpodTree:
    def __init__(self, u):
        self.u = u
        self.children = []
        self.is_last = False

    def add_node(self, u, index):
        if len(index) == 0:
            self.children.append(RpodTree(u))
            self.is_last = False
        else:
            self.children[index[0]].add_node( u, index[1:])

    def add_leaf(self, u, v, index):
        if len(index) == 0:
            self.children.append(RpodLeaf(u, v))
            self.is_last = True
        else:
            self.children[index[0]].add_leaf(u, v, index[1:])

    def __str__(self, string=''):
        list_tree = [[]]
        depth = 0
        self._rec_print(list_tree, depth)
        string = ''
        for i in range(len(list_tree)):
            for j in range(len(list_tree[i])):
                string += list_tree[i][j] + ' '
            string += "\n"
        return string

    def _rec_print(self, list_tree, depth):
        if self.is_last:
            for i in range(len(self.children)):
                # list_tree[depth].append("U: {0} V: {1}\n".format(self.children[i].u, self.children[i].v))
                list_tree[depth].append("leaf")
            list_tree[depth].append('|')
        else:
            for i in range(len(self.children)):
                list_tree[depth].append('node')
            list_tree[depth].append('|')
            if len(list_tree) == depth+1:
                list_tree.append([])
            for i in range(len(self.children)):
                self.children[i]._rec_print(list_tree, depth+1)


class RpodLeaf:
    def __init__(self, u, v):
        self.u = u
        self.v = v

from graphviz import Digraph
from pydecomp.core.RPOD import RecursiveTensor
from pydecomp.core.cls_RpodTree import RpodTree
import numpy as np

def graph_example_rpod(r):
    """Plots a typical example of rpod graph"""
    g = Digraph(filename="../plots/rpod_example",comment='RPOD graph')

    g.attr('node',shape="box",style="filled",color="lightgrey")
    root="POD(f(x1;x2,x3))"
    g.node(root)
    for r1 in range(r[0]):
        current_mode="u1[{0}], φ1[{0}]".format(r1+1)
        g.attr('node',shape="ellipse",style="empty",color="black")
        label="σ1[{0}]".format(r1+1)
        g.edge(root,current_mode,label=label)

        new_pod="POD(φ1[{0}](x2,x3))".format(r1+1)
        g.attr('node',shape="box",style="filled",color="lightgrey")
        g.edge(current_mode,new_pod)
        for r2 in range(r[1][r1]):
            g.attr('node',shape="ellipse",style="empty",color="black")
            label="σ1[{0}{1}]".format(r1+1,r2+1)
            g.edge(new_pod,"u2[{0}{1}], u3[{0}{1}]".format(r1+1,r2+1),label=label)

    g.view()

def plot_rpod_tree(rec_tens, cutoff_tol=1e-16):
    """ Plotting rpod graph """
    d=rec_tens.ndim

    g = Digraph(filename="../plots/rpod_graph",comment='RPOD graph')

    g.attr('node',shape="box",style="filled",color="lightgrey")
    root="POD(f)"
    g.node(root)
    plot_rpod_rec_tree(g,root,rec_tens.tree,0,[],cutoff_tol)
    g.view()
    return

def plot_rpod_rec_tree(graph, upper_node, tree, depth,pos, cutoff_tol=1e-16):
    r=0
    for child in tree.children:
        r+=1
        pos.append(r)
        index="".join(map(str,pos))
        edge_width=str(0.7*np.log(1000*child.branch_weight))
        label="{:2.3f}".format(child.branch_weight)
        graph.attr('node',shape="ellipse",style="empty",color="black")
        if child.branch_weight>cutoff_tol or r==1:
            if tree.is_last:
                current_mode="u{1}[{0}], u{2}[{0}]".format(index,depth+1,depth+2)
                graph.edge(upper_node,current_mode,label=label,penwidth=edge_width)
            else:
                current_mode="u{1}[{0}], φ{1}[{0}]".format(index,depth+1)
                graph.edge(upper_node,current_mode,label=label,penwidth=edge_width)

                new_pod="POD(φ{1}[{0}](x{1},w))".format(index,depth+1)
                graph.attr('node',shape="box",style="filled",color="lightgrey")
                graph.edge(current_mode,new_pod,penwidth=edge_width)
                plot_rpod_rec_tree(graph,new_pod,child,depth+1,pos,cutoff_tol)
        pos.pop()
    return

if __name__=="__main__":
    # graph_example_rpod([3,[3,2,1]])
    from benchmark_multivariable import benchmark_multivariable
    decomp_methods=["RPOD"]
    solver=["trapezes"]
    approx=benchmark_multivariable(decomp_methods, solver ,shape=[32,32,32,32,32],
                                test_function=2, plot=False,output_decomp='',
                                plot_name='',tol=1e-5)
    plot_rpod_tree(approx, cutoff_tol=1e-16)
    print(approx)

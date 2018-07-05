import matplotlib.pyplot as plt
import numpy as np

def mode_1D_plot(modes_dict,show=True,plot_name=None):
    """This function plots 1D modes and gives PDF and plot"""
    styles=['r+-','b*-','ko-','gh:','mh--']
    fig=plt.figure()
    xlim=[0,0]
    ylim=[0.1,0.1]
    k=0
    plt.yscale('lin')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()

    for label, data in approx_data.items():
        if not grid:
            grid=np.arange(0,1,data[:,0].size)
        for i in range(1):
            mode=data[:,0]
            ax=fig.add_subplot(111)
            plt.plot(grid, mode , styles[k], label=label)
        k+=1
    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    if plot_name:
        fig.savefig(plot_name)
    plt.close()
    return


def rank_benchmark_plotter(approx_data, show=True, plot_name="plots/benchmark.pdf",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles=['r+-','bx--','k8-','gs--','m3--']
    fig=plt.figure()
    xmax=1
    ylim=[0.1,0.1]
    k=0
    plt.yscale('log')
    plt.xlabel("rank")
    plt.ylabel('Relative Error')
    plt.grid()

    for label, err in approx_data.items():
        ranks=np.arange(1,err.size+1)
        xmax=max(xmax,ranks[-1])
        ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
        ax=fig.add_subplot(111)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ylim)
        plt.plot(ranks, err , styles[k], label=label)

        k+=1
    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name)
    plt.close()

import matplotlib.pyplot as plt
import numpy as np

def mode_1D_plot(modes_dict,show=True,plot_name=None):
    """This function plots 1D modes and gives PDF and plot"""

    font = {'family' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    plt.rc('legend', fontsize=12)
    linestyle={'linewidth':2,'markersize':6, 'markeredgewidth':2}
    fig=plt.figure(figsize=(7,6))
    styles=['r1-','b1--','r2-','b2--','r3-','b3--','r4-','b4-']
    xlim=[0,0]
    ylim=[0.1,0.1]
    k=0
    plt.xlabel("x")
    plt.ylabel('Relative Error')
    plt.grid()
    def_grid=False
    for i in range(modes_dict['PGD'].shape[1]):
        correlation=modes_dict['PGD'][:,i].T @ modes_dict['POD'][:,i]
        modes_dict['PGD'][:,i]=modes_dict['PGD'][:,i]/correlation
    for label, data in modes_dict.items():
        for i in range(data.shape[1]):
            mode=data[:,i]
            if not def_grid:
                grid=np.linspace(0,1,mode.size)
                def_grid=True
            ax=fig.add_subplot(111)
            plt.plot(grid, mode , styles[k], label=label+" Y_{}".format(i+1))
            # plt.plot(grid, mode , styles[k], label=label)
            k+=1
    #saving plot as pdf
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    # plt.legend()
    if show:
        plt.show()
    if plot_name:
        fig.savefig(plot_name)
    plt.close()
    return

def simple_1D_plot(modes,grid,x_label="X",show=True,plot_name=None):
    """This function plots 1D modes and gives PDF and plot"""

    font = {'family' : 'normal',
        'size'   : 14}

    plt.rc('font', **font)
    plt.rc('legend', fontsize=12)
    linestyle={'linewidth':2,'markersize':6, 'markeredgewidth':2}
    fig=plt.figure(figsize=(7,6))
    styles=['r-','b-','g-','k-','c-','m-','r4-','b4-']
    xlim=[0,0]
    ylim=[0.1,0.1]
    k=0
    plt.xlabel(x_label)
    plt.ylabel('')
    plt.grid()
    print(modes.shape)
    print(grid.size)
    for i in range(modes.shape[1]):
        mode=modes[:,i]
        ax=fig.add_subplot(111)
        plt.plot(grid, mode , label="mode {}".format(i+1))
        # plt.plot(grid, mode , styles[k], label="mode {}".format(i+1))
        k+=1
    #saving plot as pdf
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    # plt.legend()
    if show:
        plt.show()
    if plot_name:
        fig.savefig(plot_name,bbox_inches='tight')
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
    font = {'family' : 'normal',  'size'   : 13}

    plt.rc('font', **font)
    linestyle={'linewidth':2,'markersize':6, 'markeredgewidth':2}
    plt.rc('legend', fontsize=12)
    styles=['r+-','bx--','k1-','g2--','m3--']
    fig=plt.figure(figsize=(7,6))
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
        plt.plot(ranks, err , styles[k],**linestyle, label=label)

        k+=1
    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name)
    plt.close()


def benchmark_plotter(approx_data, show=True, plot_name="",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles={"HO_POD":'r+--',
            "HO_SVD":'r+:',
            "SHO_POD":'g+-',
            "ST_HOSVD":'g*-',
            "STHO_SVD":'g*-',
            "SHO_SVD":'g*-',
            "RPOD":'b+-',
            "RSVD":'b*-',
            "TT_POD":"k+-",
            "TT_SVD":'k*-',
            "QTT_SVD":'c+:',
            "PGD":'mh--',
            "SHO_SVD vectorized":'r+-',
            "SHO_SVD reshaped":'ro--',
            "TT_SVD vectorized":'k+-',
            "TT_SVD reshaped":'ko--',
            "HT":'b+-'}
    fig=plt.figure()
    xmax=0.1
    ylim=[0.1,0.1]
    k=0
    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()

    font = {'family' : 'normal',  'size'   : 13}
    plt.rc('font', **font)
    linestyle={'linewidth':2}
    plt.rc('legend', fontsize=12)

    title=kwargs.get('title',None)
    if title:
        plt.title(title)

    for label, data in approx_data.items():
        err=data[0,:]
        comp_rate=100*data[1,:]
        xmax=max(xmax,comp_rate[-1])
        ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
        ax=fig.add_subplot(111)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ylim)
        plt.plot(comp_rate, err , styles[label], **linestyle, label=label)

    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

def benchmark_norm_plotter(approx_data, show=True, plot_name="",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles={"L1":'r+-',
            "L2":'b+--',
            "Linf":'k+-'}
    fig=plt.figure()
    xmax=0.1
    ylim=[0.1,0.1]
    k=0
    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()

    font = {'family' : 'normal',  'size'   : 13}
    plt.rc('font', **font)
    linestyle={'linewidth':2}
    plt.rc('legend', fontsize=12)

    title=kwargs.get('title',None)
    if title:
        plt.title(title)
    else:
        plt.title("Norm error comparison")

    print(approx_data[1])
    comp_rate=100*approx_data[1]
    for label, data in approx_data[0].items():
        err=np.asarray(data)
        xmax=max(xmax,comp_rate[-1])
        ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
        ax=fig.add_subplot(111)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ylim)
        plt.plot(comp_rate, err , styles[label], **linestyle, label=label)

    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close()

def exp_data_decomp_plotter(approx_data, show=True, plot_name="",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles={"density":'r+-',
            "vorticity":'k+-',
            "pressure":'g+-',
            "velocity_u":'b*-',
            "velocity_v":'b+-'
            }

    fig=plt.figure()
    xmax=0.1
    ylim=[0.1,0.1]
    k=0
    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()

    font = {'family' : 'normal',  'size'   : 13}
    plt.rc('font', **font)
    linestyle={'linewidth':2}
    plt.rc('legend', fontsize=12)

    title=kwargs.get('title',None)
    if title:
        plt.title(title)

    for label, data in approx_data.items():
        err=data[0,:]
        comp_rate=100*data[1,:]
        xmax=max(xmax,comp_rate[-1])
        ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
        ax=fig.add_subplot(111)
        ax.set_xlim(0,xmax)
        ax.set_ylim(ylim)
        plt.plot(comp_rate, err , styles[label], **linestyle, label=label)

    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name)
    plt.close()

def several_d_plotter(approx_data, show=True, plot_name="",**kwargs):
    """
    Plotter routine for benchmark function.

    **Parameters**:
    *approx_data* [dict] whose labels are the lines labels, and data is
                         a 2-column array with first column compression rate,
                         second one is the approximation error associated.
    *show* [bool]        whether the plot is shown or not
    *plot_name* [str]    plot output location, if empty string, no plot
    """
    styles={"HO_POD":'g',
            "SHO_POD":'r',
            "RPOD":'b',
            "HO_SVD":'g',
            "SHO_SVD":'r',
            "RSVD":'b',
            "TT_SVD":'k',
            "PGD":'m'}
    linestyles={2:"-",
               3:":",
               4:"--",
               5:"-",
               6:"-",
               7:"--",
               8:":",
               9:"-"}

    fig=plt.figure()
    xmax=0.1
    ylim=[0.1,0.1]
    k=0
    font = {'family' : 'normal',  'size'   : 13}
    plt.rc('font', **font)
    linestyle={'linewidth':2,'markersize':6, 'markeredgewidth':2}
    plt.rc('legend', fontsize=12)
    fig=plt.figure(figsize=(7,6))

    plt.yscale('log')
    plt.xlabel("Compresion rate (%)")
    plt.ylabel('Relative Error')
    plt.grid()

    title=kwargs.get('title',None)
    if title:
        plt.title(title)
    for d,ddata in approx_data.items():
        ls=linestyles[d]
        for label, data in ddata.items():
            err=data[0,:]
            comp_rate=100*data[1,:]
            xmax=max(xmax,comp_rate[-1])
            ylim=[min(ylim[0],err[-1]),max(ylim[1],err[0])]
            ax=fig.add_subplot(111)
            ax.set_xlim(0,xmax)
            ax.set_ylim(ylim)
            lc=styles[label]
            plt.plot(comp_rate, err ,lc+ls ,**linestyle, label=label+"_{}".format(d))

    #saving plot as pdf
    plt.legend()
    if show:
        plt.show()
    fig.savefig(plot_name)
    plt.close()

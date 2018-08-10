#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 9 12:00:00 2018

@author: Lucas

Module that reads, compress and analyse data from Tapan LDC simulation.
"""
import os,sys
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm

from core.Tucker import tucker_error_data, TuckerTensor
from core.tucker_decomp import STHOSVD
from core.TensorTrain import TensorTrain, error_TT_data
from core.TT_SVD import TT_SVD

from analysis.plot import benchmark_plotter


def LDC_multi_Re_decomp(path,Re_list,layouts=['reshaped','vectorized'],
                        tol=1e-6,show_plot=True,plot_name=""):
    file='LDC_binary_Re_{}.dat'.format(Re_list)
    if file in os.listdir(path):
        Tensor=load(path+file)
    else:
        Tensor=read_ldc_data_multiple_Re(path,Re_list)
        save(Tensor, file_name=path+file)
    shape=Tensor.shape
    print('Read LDC tensor of shape:', shape)

    T_approx={}
    approx_data={}
    for layout in layouts:
        if layout == 'reshaped':
            nx=ny=np.sqrt(shape[0])
            T_full=np.reshape(Tensor,(nx,ny,shape[1],shape[2]))
            key=' reshaped'
        elif layout == "vectorized":
            T_full=Tensor
            key=' vectorized'
        else :
            AttributeError("Invalid layour option : {}".format(layout))

        method="SHO_SVD"+key
        T_approx[method]=STHOSVD(T_full,epsilon=tol)
        print("{} Rank ST-HOSVD decomposition with tol {}".format(T_approx[method].core.shape,tol))
        approx_data[method]=np.stack(tucker_error_data(T_approx[method],T_full))

        method="TT_SVD"+key
        T_approx[method]=TT_SVD(T_full,eps=tol)
        print("{} Rank TT decomposition with tol {}".format(T_approx[method].rank,tol))
        approx_data[method]=np.stack(error_TT_data(T_approx[method],T_full))

    if show_plot or (plot_name!=""):
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,
                        title="LDC data decomposition error")



def read_ldc_data_multiple_Re(path, Re_list):
    vort_tensor=[]
    for Re in Re_list:
        w,t_grid,nx,nt=read_ldc_data(path+"LDC_{}/".format(Re))
        vort_tensor.append(w)
    return np.stack(vort_tensor,axis=2)


def read_ldc_data(path):
    """
    This function reads the data  (unzipped) yielded by Tapan's LDC code and loads
    it to a 2 way array.
    Files are expected to be in the following format:
        'svt_nnnn.nnnnnn.dat' where n is a digit
    input
        path is the folder of interest.
    output
        w       A 2 way ndarray
    """
    time_list=[]
    for entry in os.listdir(path):
        if entry.startswith('svt_') and entry.endswith('.dat'):
            try:
                t = float(entry[4:-5])
                time_list.append([entry,t])
            except:
                print('file '+entry+' was discarded.')
                continue
    time_list.sort()
    buff = read_file(path,time_list[0][0])
    w=np.zeros([buff.shape[0],len(time_list)])
    t_grid=np.zeros(len(time_list))

    for i in range(len(time_list)):
        filename=time_list[i][0]
        print('importing '+path+filename)
        t_grid[i]=time_list[i][1]
        buff=read_file(path,filename)
        w[:,i]=buff[:,3].copy()
    nx=w.shape[0]
    nt=w.shape[1]
    return w, t_grid, nx,nt

def read_file(folder,filename , sep='\t') :
    """ This function reads the data yielded by Tapan's DNS at a given point i.e. file
        folder      is the folder in which the data is to be read
        filename    is the name of the file
        return data a 2D array that contains of dim (Nt= number of files, Nx = number of lines -2 )
    """
    full_address = folder+filename
    #print("full adress is : \n",full_address)
    try:
        flux = open(full_address, mode='r')
    except:
        print("wrong file name :'"+full_address+"'")
        sys.exit()
    M=[]
    for line in flux:
        try:
            X = [float(x) for x in line.split()]
            M.append(X)
        except:
            #print("this line does not contain floats only <"+line.strip()+">")
            continue

    flux.close()
    return np.squeeze(np.asanyarray(M))



################################################################################
################################################################################
#                              PLOTTING PART
################################################################################
################################################################################

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.colors import LogNorm
import numpy as np

def plot_spatial_interp(phi_exact,phi_interp,target_Re,nb_modes=1,path='screen',
                        PLOT_PHI=True,PLOT_ERR=True):
    """ This function returns various plots of phi, phi_exact, error...
        PARAMETERS:
            -phi_exact       2-way array of the exact solution spatial modes
            -phi_interp      2-way array with the interpolated phis
            -nb_modes        the number of modes to be plotted
            -target_Re       Value of the target Re for naming only
            -path='screen'   where the plot is sent, if screen, on screen,
                             otherwise, to the given location
            -PLOT_PHI=True   If true, phi interp is plotted
            -PLOT_ERR=True   If true, Interpolation error is plotted
        returns
            void
    """
    if phi_exact.ndim==1:
        loc_err=np.zeros((phi_exact.size,1))
        loc_err[:,0]=np.abs(phi_exact-phi_interp)
    else:
        loc_err=np.abs(phi_exact[:,:nb_modes]-phi_interp[:,:nb_modes])


    min_contour=-0.3
    max_contour=0.3
    n_contour=51
    x=y=np.linspace(0,1,257)
    X, Y = np.meshgrid(x, y)
    print(loc_err.shape)
    if path=='screen':
        origin='lower'
        plt.pcolor(X, Y, loc_err[:,0].reshape([257,257]) ,
                   norm=LogNorm(vmin=1e-3, vmax=100),
                   cmap='gnuplot')
        plt.colorbar()
        plt.title('Error map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.figure()

        CS = plt.contourf(X, Y, phi_interp[:].reshape([257,257]),
                          levels=np.linspace(min_contour,max_contour,n_contour),
                          cmap=plt.cm.gnuplot,
                          origin=origin)

        CS2 = plt.contour(CS, levels=CS.levels,
                          colors='k',
                          origin=origin,
                          extend='both')

        CS2.cmap.set_under('blue')
        CS2.cmap.set_over('red')

        plt.title('Vorticity contour interpolated mode')
        plt.xlabel('X')
        plt.ylabel('Y')

        cbar = plt.colorbar(CS)
        cbar.ax.set_ylabel('verbosity coefficient')
        cbar.add_lines(CS2)
        plt.show()
        plt.figure()
        print("blabla")

        CS3 = plt.contourf(X, Y, phi_exact[:].reshape([257,257]),
                          levels=np.linspace(min_contour,max_contour,n_contour),
                          cmap=plt.cm.gnuplot,
                          origin=origin)

        CS4 = plt.contour(CS3, levels=CS3.levels,
                          colors='k',
                          origin=origin,
                          extend='both')


        plt.title('Vorticity contour exact mode')
        plt.xlabel('X')
        plt.ylabel('Y')

        cbar = plt.colorbar(CS3)
        cbar.ax.set_ylabel('verbosity coefficient')
        cbar.add_lines(CS4)
        plt.show()
        plt.figure()


    else:

        pp = PdfPages(path+'phi_interp_'+str(target_Re)+'.pdf')
        origin='lower'
        for i in range(nb_modes):
            glob_err=np.linalg.norm(loc_err[:,i])/np.linalg.norm(phi_exact[:,i])
            print("Global error on mode "+str(i)+" is "+str(glob_err))
            plt.figure(figsize=(7,18))
            plt.subplot(311)
            plt.pcolor(X, Y, loc_err[:,i].reshape([257,257]) ,
                       norm=LogNorm(vmin=1e-3, vmax=1e2),
                       cmap='bwr')
            plt.colorbar()
            plt.title('Error map mode '+str(i+1)+', global error ='+str(glob_err))
            plt.xlabel('X')
            plt.ylabel('Y')

            plt.subplot(312)
            CS = plt.contourf(X, Y, phi_interp[:,i].reshape([257,257]),
                          levels=np.linspace(min_contour,max_contour,n_contour),
                              cmap=plt.cm.gnuplot,
                              origin=origin)

            CS2 = plt.contour(CS, levels=CS.levels,
                              colors='k',
                              origin=origin,
                              extend='both')

            plt.title('Vorticity contour interpolated mode')


            cbar = plt.colorbar(CS)
            cbar.ax.set_ylabel('verbosity coefficient')
            cbar.add_lines(CS2)

            plt.subplot(313)
            CS3 = plt.contourf(X, Y, phi_exact[:,i].reshape([257,257]),
                          levels=np.linspace(min_contour,max_contour,n_contour),
                               cmap=plt.cm.gnuplot,
                               origin=origin)

            CS4 = plt.contour(CS3, levels=CS3.levels,
                              colors='k',
                              origin=origin,
                              extend='both')

            plt.title('Vorticity contour exact mode')
            cbar = plt.colorbar(CS3)
            cbar.ax.set_ylabel('verbosity coefficient')
            cbar.add_lines(CS4)

            plt.savefig(pp,format='pdf')
            plt.close()
            print('visu of spatial mode '+str(i)+' complete')

        pp.close()

def plot_spatial_modes(phi,path='screen',output_name='phi_range.pdf',min_contour=-0.3, max_contour=0.3,n_contour=51):
    """ This function returns various plots of phi, phi_exact, error...

        PARAMETERS:
            -phi             2-way array of the exact solution spatial modes
            -path='screen'   where the plot is sent, if screen, on screen, otherwise, to the given location
            -output_name     name of the output file
            -min_contour     minimum contour value
            -max_contour     maximum contour value
            -n_contour       number of contour lines
        returns
            void
    """
    print(np.shape(phi))
    #nx = ny= int(np.sqrt(phi.shape[1]))
    nx = ny= int(np.sqrt(phi.shape[1]))
    print(nx,ny)
    x=np.linspace(0,1,nx)
    y=np.linspace(0,1,ny)
    X, Y = np.meshgrid(x, y)
    nb_modes=np.size(phi[:,0])
    if path!='screen':
        pp = PdfPages(path+output_name)
        plt.figure(figsize=(7,9))
    origin='lower'
    lev=np.linspace(min_contour,max_contour,n_contour)

    for i in range(2,nb_modes):
        CS = plt.contourf(X, Y, phi[i,:].reshape([nx,ny]),
                          levels=lev,
                          cmap=plt.cm.seismic,
                          origin=origin)

        CS2 = plt.contour(CS, levels=CS.levels,
                          colors='k',
                          origin=origin,
                          extend='both')

        plt.title('Vorticity contour mode')
        plt.xlabel('X')
        plt.ylabel('Y')

        cbar = plt.colorbar(CS)
        cbar.add_lines(CS2)
        plt.show()
        if path=='screen':
            plt.figure()
        else:
            plt.savefig(pp,format='pdf')
            plt.close()

        print('visu of spatial mode '+str(i)+' complete')

    if path!='screen':
        pp.close()

    return

def plot_vorticity_exponential_contour(w,path='screen',output_name='vorticity_contour_plot.pdf'
                                        ,n_contour=51,centered=False,t='1900'):
    """ This function returns various plots of phi, phi_exact, error...

        PARAMETERS:
            -w               2-way array output of LDC code
            -path='screen'   where the plot is sent, if screen, on screen, otherwise, to the given location
            -output_name     name of the output file
            -n_contour       number of contour lines
        returns
            void
    """
    print(np.shape(w))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    nx = ny= int(np.sqrt(w.shape[0]))
    print(nx,ny)
    x=np.linspace(0,1,nx)
    y=np.linspace(0,1,ny)
    X, Y = np.meshgrid(x, y)
    if path!='screen':
        pp = PdfPages(path+output_name)
        fig=plt.figure(figsize=(7,5))
    else:
        fig=plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    origin='lower'
    nlCenter=2000#33027 #line number of the central point of the grid
    w_center=w[nlCenter,3]
    w_max=w.max()
    w_min=w.min()
    if (centered) :
        w[:,3]=w[:,3]-w_center
        print("cetered to ",w_center)
        #w_center=0
    grid=np.exp(np.power(np.linspace(0,1.4,n_contour),3))-1
    grid_neg=-grid
    print(type(grid))
    lev= np.zeros(2*grid.size-1)
    lev[:n_contour]=grid_neg[::-1]
    lev[n_contour-1:]=grid
    if (not centered):  lev=lev+w_center
    print(lev)
    print(lev.size)
    print(nx,ny)

    tick=[w_min]
    tick=np.append(tick,lev[0:-1:2])
    tick=np.append(tick,[w_max])
 #   lev=np.linspace(0,1,10)
    CS = plt.contourf(X,Y, w[:,3].reshape([nx,ny]),
                      levels=lev,
                      cmap=plt.cm.jet,spacing='proportional',
                      origin=origin,boundaries=tick,
                      extend='both',ticks=tick)

    CS2 = plt.contour(CS, levels=CS.levels,
                      colors='k',spacing='proportional',
                      origin=origin,boundaries=tick,
                      extend='both',ticks=tick)
    if centered:
        plt.title('Vorticity contour, t='+t)#+' \n (w_center={:.3f}'.format(w_center)+
#                  ', min='+str(int(w_min))+', max='+str(int(w_max))+')')
    else:
        plt.title('Centered vorticity contour, t='+t)#+' \n ('+
                  #', min='+str(int(w_min))+', max='+str(int(w_max))+')')

    plt.xlabel('X')
    plt.ylabel('Y',rotation=0)

    cbar = plt.colorbar(CS,pad=0.1)
    cbar.add_lines(CS2)
    cbar.set_label('Vorticity level')
    #cbar.set_ticklabels()
    tick=[w_min]
    tick=np.append(tick,lev[0:-1:2])
    tick=np.append(tick,[w_max])
    tick=lev[::3]
    #setting ticks positions
    cbar.set_ticks(tick)
    #offset of the value to the real value of w
    #tick+=w_center

    tick=['{:.3f}'.format(x+w_center) for x in tick]
    cbar.set_ticklabels(tick)
    ax.annotate("C",xy=(0.5,0.5), xytext=(0.465, 0.46))#,arrowprops=dict( color='black',arro))
    ax.annotate("",xy=(0.485,0.5), xytext=(0.515, 0.5),arrowprops=dict( color='black',arrowstyle='-'))
    ax.annotate("",xy=(0.5,0.485), xytext=(0.5, 0.515),arrowprops=dict( color='black',arrowstyle='-'))
    ax.annotate('{:.0f}'.format(w_max), xy=(1, 0), xytext=(1.12, 1.01))
    ax.annotate('{:.0f}'.format(w_min), xy=(1, 0), xytext=(1.115, -0.04))
    ax.annotate(r"$\omega_c$", fontsize=20, xy=(1, 0.5), xytext=(1.06,0.48))
    cbar.ax.annotate("", xy=(-0., 0.5), xytext=(1, 0.5),
            arrowprops=dict( color='red')
            )
#    cbar.ax.
    plt.show()
    if path!='screen':
        plt.savefig(pp,format='pdf')
        plt.close()

    print('visu of vorticity field complete')

    if path!='screen':
        pp.close()

    return

if __name__ == "__main__":
    from utils.IO import save, load
    from time import time
    # path_dns="/home/lestandi/Documents/data_ldc/grid_257x257/LDC_9800/"
    # n_levels=16
    # w = read_file(path_dns,'svt_1939.999000.dat')
    # plot_vorticity_exponential_contour(w,path="/home/lestandi/Bureau/",
    #                                     output_name='vort_contour.pdf',
    #                                     n_contour=n_levels, centered=True,
    #                                     t='1900.2')
    # path='/home/lestandi/Documents/data_ldc/grid_257x257/LDC_10000/'
    # w, t_grid, nx, nt=read_ldc_data(path)
    # print(w.shape)
    Re_list=[10000,10020,10040,10060,10080,10100]
    path='/home/lestandi/Documents/data_ldc/grid_257x257/'
    layouts=["vectorized",'reshaped']
    LDC_multi_Re_decomp(path,Re_list,layouts,tol=1e-16,show_plot=True,
                        plot_name="../output/LDC_compr_data/decomp_error_graph.pdf")

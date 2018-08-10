#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:17:31 2018

@author: diego
"""

import numpy as np
import h5py
import core.tensor_algebra as ta

from core.Tucker import tucker_error_data, TuckerTensor
from core.tucker_decomp import STHOSVD
from core.TensorTrain import TensorTrain, error_TT_data
from core.TT_SVD import TT_SVD

from analysis.plot import benchmark_plotter, simple_1D_plot

def matlab_file_reduction(file_name, tol=1e-4,
                          show_plot=True, plot_name=""):
    '''
    A special tensor decomposition for C. Pradère matlab file.
    '''
    T_full=matlab_file_to_tensor(file_name,False)
    print("Matlab Imput data shape is ", T_full.shape)
    T_approx={}
    approx_data={}

    T_approx["SHO_SVD"]=STHOSVD(T_full,epsilon=tol)
    print("{} Rank ST-HOSVD decomposition with tol {}".format(T_approx["SHO_SVD"].core.shape,tol))
    T_approx["TT_SVD"]=TT_SVD(T_full,eps=tol)
    print("{} Rank TT decomposition with tol {}".format(T_approx["TT_SVD"].rank,tol))

    approx_data["SHO_SVD"]=np.stack(tucker_error_data(T_approx["SHO_SVD"],T_full))
    approx_data["TT_SVD"]=np.stack(error_TT_data(T_approx["TT_SVD"],T_full))

    T_full=np.reshape(T_full,(29,51,-1))
    T_approx["SHO_SVD vectorized"]=STHOSVD(T_full,epsilon=tol)
    print("{} Rank ST-HOSVD decomposition with tol {}".format(T_approx["SHO_SVD vectorized"].core.shape,tol))
    T_approx["TT_SVD vectorized"]=TT_SVD(T_full,eps=tol)
    print("{} Rank TT decomposition with tol {}".format(T_approx["TT_SVD"].rank,tol))



    if show_plot or (plot_name!=""):
        approx_data["SHO_SVD vectorized"]=np.stack(tucker_error_data(T_approx["SHO_SVD vectorized"],T_full))
        approx_data["TT_SVD vectorized"]=np.stack(error_TT_data(T_approx["TT_SVD vectorized"],T_full))

        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,
                        title="Lab exprement data: evaporating droplets")

    simple_1D_plot(T_approx["SHO_SVD"].u[1][:,:5], np.arange(0,51),
                   x_label="Wavelength", show=True,
                   plot_name="../output/exp_data/lambda_modes_plot.pdf")

    simple_1D_plot(T_approx["SHO_SVD"].u[0][:,:5], np.arange(0,29),
                   x_label="time", show=True,
                   plot_name="../output/exp_data/time_modes_plot.pdf")

    return

def matlab_file_to_tensor(file_name,vectorize=False):
    """ A special reader for C. Pradère matlab file """
    f=h5py.File(file_name,'r')
    data=np.array(f[list(f.keys())[0]])
    nt=data.shape[0]
    nl=data.shape[1]
    if vectorize:
        data=np.reshape(data,(nt,nl,-1))
    return data

def matlab_data_view(tensor,t,freq,show_plot=True,file_name=""):
    import matplotlib.pyplot as plt
    slice=tensor[t,freq,:,:]

    if file_name != "":
        plt.figure(figsize=(7,6))
    plt.imshow(slice, cmap=plt.cm.gray)
    if show_plot:
        plt.show()
    if file_name != "":
        plt.savefig(file_name, bbox_inches='tight',pad_inches=0)
        plt.close()

def matlab_data_view_animate(tensor,wavelength,t_min,t_max):
    """ This routine provides an animated view (gif) of the matlab data files at
    wavelength between t_min and t_max
    """
    from utils.misc import quick_gif
    dir="../output/exp_data/visu"
    for l in wavelength:
        plt_list=[]
        for t in range(t_min, t_max):
            plt_name=dir+"/droplet_{}_{}.png".format(l,t)
            plt_list.append(plt_name)
            matlab_data_view(tensor,t,l,show_plot=False,file_name=plt_name)
        quick_gif(plt_list,dir,'droplet_animate_lambda_{}.gif'.format(l))
    return

if __name__=='__main__':
    path_data="../exp_data/Exemple_1.mat"
    matlab_file_reduction("../exp_data/Exemple_1.mat",tol=1e-6, show_plot=True,
                          plot_name="../output/exp_data/matlab_pradere.pdf")
                          # plot_name="")

    # tensor=matlab_file_to_tensor(path_data)
        # matlab_data_view(tensor,0,0,show_plot=False,file_name="../output/exp_data/droplet_view_test.png")
    # freq=[0,20,21,50]
    # t_min=0
    # t_max=29
    # matlab_data_view_animate(tensor,freq,t_min,t_max)

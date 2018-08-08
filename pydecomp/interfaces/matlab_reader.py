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

from analysis.plot import benchmark_plotter

def matlab_file_reduction(file_name, tol=1e-4,
                          show_plot=True, plot_name=""):
    '''
    A special tensor decomposition for C. Pradère matlab file.
    '''
    T_full=matlab_file_to_tensor(file_name)
    print("Matlab Imput data shape is ", T_full.shape)
    T_approx={}
    approx_data={}

    T_approx["SHO_SVD"]=STHOSVD(T_full,epsilon=tol)
    print("{} Rank ST-HOSVD decomposition with tol {}".format(T_approx["SHO_SVD"].core.shape,tol))
    approx_data["SHO_SVD"]=np.stack(tucker_error_data(T_approx["SHO_SVD"],T_full))

    T_approx["TT_SVD"]=TT_SVD(T_full,eps=tol)
    print("{} Rank TT decomposition with tol {}".format(T_approx["TT_SVD"].rank,tol))
    approx_data["TT_SVD"]=np.stack(error_TT_data(T_approx["TT_SVD"],T_full))

    if show_plot or (plot_name!=""):
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,
                        title="Lab exprement data: evaporating droplets")
    return

def matlab_file_to_tensor(file_name):
    """ A special reader for C. Pradère matlab file """
    f=h5py.File(file_name,'r')
    data=f[list(f.keys())[0]]
    return np.array(data)


if __name__=='__main__':
    matlab_file_reduction("../exp_data/Exemple_1.mat",tol=1e-6, show_plot=True, plot_name="../output/exp_data/")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 7 10:38:16 2018

@author: Lucas

This module encompasses function for decomposition of real data.
"""
from interfaces.bp_compressor import *
from interfaces.tapan_LDC_decomp import LDC_multi_Re_decomp
from interfaces.matlab_reader import matlab_file_reduction

def test_notus_wave_bp(cases):
    """
    Short script for numerical tests on notus wave data stored in bp format
    Two kinds of decompositions are provides
    -'distinct_decomp' : decomposition is performed separately on each variable,
                        a comparison between each variable is proposed.
    -'variables_as_dim': the variables are put together to form a new,
                        a comparison between each variable is proposed.
    """
    var_list=['density','pressure','vorticity','velocity_u','velocity_v']
    data_dir="../data_notus_wave_small/"
    data_dir="../data_notus_wave/"
    out_dir='../output/compressed_notus_wave/'

    if 'distinct_decomp' in cases:
        bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-2,rank=-1 )
        Analize_compressed_bp(bp_compressed_out,show_plot=True, plot_name=out_dir+"ST_HOSVD_wave_plot.pdf")

    if 'variables_as_dim' in cases:
        bp_compressed_out= bp_compressor_variables_as_dim(var_list,data_dir, tol=1e-16,rank=-1 )
        Analize_compressed_bp_vars_as_dim(bp_compressed_out,show_plot=True,
                                          plot_name=out_dir+"wave_decomposition_plot.pdf"
                                          ,variables=var_list)

    return

def test_LDC_data():
    """ Decomposition of Tapan LDC data: a lot is still to be done ! """

    Re_list=[10000,10020,10040,10060,10080,10100]
    path='/home/lestandi/Documents/data_ldc/grid_257x257/'
    layouts=["vectorized",'reshaped']
    LDC_multi_Re_decomp(path,Re_list,layouts,tol=1e-6,show_plot=True,
                        plot_name="../output/LDC_compr_data/decomp_error_graph.pdf")

    data_file='LDC_binary_Re_{}.dat'.format(Re_list)
    decomp_path="../output/LDC_compr_data/compressions_dict.dat"
    data_path=path+data_file
    plot_path="../output/LDC_compr_data/"
    spatial_plotting_for_manuscript(T_full_path, T_approx_path, plot_path)

    return

def test_matlab_data():
    """ Decomposition of C. Pradère data: Adapt Diego's existing work ! """
    data_path="../exp_data/Exemple_1.mat"
    plot_name="../output/exp_data/matlab_pradere.pdf"
    matlab_file_reduction(data_path,tol=1e-8, show_plot=True,
                          plot_name=plot_name)
    return

if __name__=="__main__":
    cases=['variables_as_dim','distinct_decomp']
    cases=[cases[1]]
    test_notus_wave_bp(cases)

    # test_LDC_data()

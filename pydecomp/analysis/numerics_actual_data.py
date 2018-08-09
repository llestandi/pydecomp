#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 7 10:38:16 2018

@author: Lucas

This module encompasses function for decomposition of real data.
"""
from interfaces.bp_compressor import *
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
        bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-6,rank=-1 )
        Analize_compressed_bp(bp_compressed_out,show_plot=False, plot_name=out_dir+"ST_HOSVD_wave_plot.pdf")

    if 'variables_as_dim' in cases:
        bp_compressed_out= bp_compressor_variables_as_dim(var_list,data_dir, tol=1e-16,rank=-1 )
        Analize_compressed_bp_vars_as_dim(bp_compressed_out,show_plot=True,
                                          plot_name=out_dir+"wave_decomposition_plot.pdf"
                                          ,variables=var_list)

    return

def test_LDC_data():
    """ Decomposition of Tapan LDC data: a lot is still to be done ! """
    return

def test_matlab_data():
    """ Decomposition of C. Pradère data: Adapt Diego's existing work ! """
    return

if __name__=="__main__":
    cases=['variables_as_dim','distinct_decomp']
    cases=['variables_as_dim']
    test_notus_wave_bp(cases)

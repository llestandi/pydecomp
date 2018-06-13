#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:38:16 2018

@author: diego
"""

from bp_reader import bp_reader
from SHOPOD import SHOPOD
import high_order_decomposition_method_functions as hf
import numpy as np

def bp_compressor(Variable,data_dir, Sim_list=0, tol=1e-7,rank=-1):
    """
    In this code we are going to compress the variable data
    contained in a bp folder using the SHOPOD as a reduction method
    with a unitary matrix as mass matrices to simulate the result of
    high order SVD method to avoid the use of the grid.\n
    **Parameter:** \n

    *Variable*: string type, the name of the variable to extract is spected
    such as "pressure", "density", "vitesse" etc. \n

    *data_dir*: string, indicates the position of the interest files

    *Sim_list*: If there are several simulations in the
    same file (simulations for differents cases) adios will create
    differents files, this code in order to first example will take only
    one of this simulations, default value will be 0 as there is only
    going to be at least one simulation. \n

    *tol*: This variable represents an error estimation that is taken from
    eigen values of the correlation Matrix, this is not the final error.
    A tolerance in the order of 1e-2 will generate results with much
    lower errors errors values.
    """

    field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list=bp_reader(Variable,data_dir)
    Mass=hf.unit_mass_matrix_creator(field[Sim_list])
    Reduced=SHOPOD(field[Sim_list],Mass,tol=tol,rank=10)

    return field[Sim_list], Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list


if __name__=='__main__':
    from evtk.hl import gridToVTK
    from output_vtk import *
    from plot_error_tucker import plot_error_tucker

    var_list="density"
    data_dir="data_notus_wave/"
    out_dir='output/'
    base_name="Lucas_HL009_dL010_"
    field,Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list=bp_compressor(var_list,data_dir, tol=1e-5 )
    print(Reduced.core.shape)
    Approx_FF=Reduced.reconstruction()
    plot_error_tucker(Reduced,field,1, 'Error vs compression rate',
                      output_variable_name='Matlab_compression_file')
    field_approx=np.copy(np.transpose(Approx_FF.tondarray()),order='F')
    var_dict={var_list:field.T,var_list+'Compressed':field_approx, "diff":field.T-field_approx}

    VTK_save_space_time_field(var_dict,X,Y,out_dir+base_name,time_list)
    print(np.linalg.norm(var_dict["diff"])/np.linalg.norm(var_dict["density"]))

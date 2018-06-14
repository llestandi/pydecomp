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

def bp_compressor(Variable,data_dir, Sim_list=-1, tol=1e-7,rank=-1):
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
    field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heigts=bp_reader(Variable,data_dir)
    print("bp reader output field", field.shape)
    if Sim_list!=-1:
        # field=field[]
        full=True


    Mass=hf.unit_mass_matrix_creator(field)

    Reduced=SHOPOD(field,Mass,tol=tol,rank=rank)

    return field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heigts

def Analize_compressed_bp(bp_compressed_out,args=[]):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed
    *args* a list of arguments for further use."""
    field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heigts=bp_compressed_out

    for i in range(Reduced.dimsize()):
        print(Reduced.u[i].shape)
    print(field.shape)

    plot_error_tucker(Reduced,field,1, 'Error vs compression rate',
    output_variable_name='bp_compression')

    print(np.linalg.norm(var_dict["diff"])/np.linalg.norm(var_dict[var_list]))
    modes={}
    for i in range(Reduced.core.shape[1]):
        modes[var_list+'_mode_'+str(i)]=Reduced.u[1][:,i]
    prepare_dic_for_vtk(modes,nxC,nyC)
    z=np.asarray([0])
    gridToVTK(out_dir+"modes",X,Y,z, cellData = modes)

def bp_comp_to_vtk(tucker_fields,X,Y,time_list,
                   full_fields=None,baseName="notus_bp_comp"):
    """Wrapper for bp decomposed fields saving to vtk format"""
    Approx_FF=Reduced.reconstruction()
    field_approx=np.copy(np.transpose(Approx_FF.tondarray()),order='F')
    var_dict={var_list:field.T,var_list+'Compressed':field_approx, "diff":field.T-field_approx}
    VTK_save_space_time_field(var_dict,X,Y,out_dir+base_name,time_list)

if __name__=='__main__':
    from evtk.hl import gridToVTK
    from output_vtk import *
    from plot_error_tucker import plot_error_tucker

    var_list="density"
    data_dir="data_notus_wave/"
    out_dir='output/'
    base_name="Lucas_HL009_dL010_"
    bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-16,rank=10 )

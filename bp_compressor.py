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

def bp_compressor(variables,data_dir, Sim_list=-1, tol=1e-7,rank=-1):
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
    field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_reader(variables,data_dir)
    Reduced={}
    for v in variables:
        vfield=field[v]
        print("field shape('"+v+"') :\t", vfield.shape)
        Mass=hf.unit_mass_matrix_creator(vfield)
        Reduced[v]=SHOPOD(vfield,Mass,tol=tol,rank=rank)

    return field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights

def Analize_compressed_bp(bp_compressed_out,**kwargs):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed
    *args* a list of arguments for further use."""

    field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_compressed_out

    plot_err=kwargs.get('plot_err',None)
    if plot_err:
        for var in Reduced:
            plot_error_tucker(Reduced[var],field[var],1, 'Error vs compression rate',
                            output_variable_name='ouptut/bp_compression')

    bp_comp_to_vtk(Reduced,X,Y,time_list,heights,
                   full_fields=field,baseName="notus_bp_comp")

    modes={}
    # for i in range(Reduced.core.shape[1]):
    #     modes[var_list+'_mode_'+str(i)]=Reduced.u[1][:,i]
    # prepare_dic_for_vtk(modes,nxC,nyC)
    # z=np.asarray([0])
    # gridToVTK(out_dir+"modes",X,Y,z, cellData = modes)

def bp_comp_to_vtk(fields_approx,X,Y,time_list, param_list,
                   full_fields=None,baseName="notus_bp_comp"):
    """
    Wrapper for bp decomposed fields saving to vtk format
    It is assumed that tucker_fields is a dictionnary of tucker format fields
    """
    print(type(param_list))
    var_dic=FT_dict_to_array(fields_approx)
    for k in range(len(param_list)):
        loc_dic={}
        for var in var_dic:
            loc_dic[var+'_decomp']=np.copy(np.transpose(var_dic[var][k,:,:]),order='F')
            try:
                loc_dic[var]=np.copy(np.transpose(full_fields[var][k,:,:]),order='F')
                loc_dic[var+'_diff']=loc_dic[var]-loc_dic[var+'_decomp']
            except:
                print('no full field')
        VTK_save_space_time_field(loc_dic,X,Y,out_dir+base_name+param_list[k]+'_',time_list)
#        print(np.linalg.norm(loc_dic[var+"diff"])/np.linalg.norm(loc_dic[var]))

    return

def CompT_dict_to_FT_dict(CompField):
    """Recontruct compressed field to full format contained in dictionnary"""
    FT_field={}
    for var in CompField:
        FT_field[var]=CompField[var].reconstruction()
    return FT_field

def FT_dict_to_array(FT_dict):
    """Recontruct compressed field to ndarray contained in dictionnary"""
    array_dict={}
    for var in FT_dict:
        print("field "+str(var)+" has a decomposition rank of "+str(FT_dict[var].core.shape))
        array_dict[var]=np.copy(FT_dict[var].reconstruction().tondarray(),order='F')
    return array_dict

if __name__=='__main__':
    from evtk.hl import gridToVTK
    from output_vtk import *
    from plot_error_tucker import plot_error_tucker

    var_list=['density']#,'pressure','vorticity','velocity_u','velocity_v']
    data_dir="data_notus_wave/"
    out_dir='output/'
    base_name="Lucas_dL010_"
    bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-6,rank=-1 )
    Analize_compressed_bp(bp_compressed_out)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:38:16 2018

@author: diego
"""

from interfaces.bp_reader import bp_reader,bp_reader_one_openning_per_file
from core.tucker_decomp import SHOPOD, STHOSVD
from core.Tucker import tucker_error_data
import core.tensor_algebra as ta
import core.MassMatrices as mm
import numpy as np
from interfaces.output_vtk import VTK_save_space_time_field

def bp_compressor(variables,data_dir, Sim_list=-1, tol=1e-4,rank=-1):
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
        MM=mm.MassMatrices([mm.identity_mass_matrix(x) for x in list(vfield.shape)])
        Reduced[v]=SHOPOD(vfield,MM,tol=tol,rank=rank)

    return field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights,MM


def Analize_compressed_bp(bp_compressed_out, show_plot=True, plot_name=""):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed"""
    from analysis.plot import exp_data_decomp_plotter
    field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights,MM=bp_compressed_out
    approx_data={}
    if show_plot or (plot_name!=""):
        for var in Reduced:
            print(Reduced[var])
            tucker_decomp=Reduced[var]
            F=field[var]
            approx_data[var]=np.stack(tucker_error_data(
                                        tucker_decomp,F,int_rules=MM))
            exp_data_decomp_plotter(approx_data, show_plot, plot_name=plot_name,
                              title="Wave simulation ST-HOPOD decomposition")

    bp_comp_to_vtk(Reduced,X,Y,time_list,heights,
                   full_fields=field,base_name="notus_bp_comp")


def bp_comp_to_vtk(fields_approx,X,Y,time_list, param_list,
                   full_fields=None,base_name="notus_bp_comp"):
    """
    Wrapper for bp decomposed fields saving to vtk format
    It is assumed that tucker_fields is a dictionnary of tucker format fields
    """
    print(type(param_list))
    out_dir="../output/compressed_notus_wave/"
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
        array_dict[var]=np.copy(FT_dict[var].reconstruction(),order='F')
    return array_dict


def bp_compressor_variables_as_dim(variables,data_dir, tol=1e-4,rank=-1):
    """
    In this code we are going to compress the variable data
    contained in a bp folder using the STHOPOD as a reduction method
    with a unitary matrix as mass matrices to simulate the result of
    high order SVD method to avoid the use of the grid.\n
    **Parameter:** \n

    *Variable*: string type, the name of the variable to extract is spected
    such as "pressure", "density", "vitesse" etc. \n

    *data_dir*: string, indicates the position of the interest files

    *tol*: This variable represents an error estimation that is taken from
    eigen values of the correlation Matrix, this is not the final error.
    A tolerance in the order of 1e-2 will generate results with much
    lower errors errors values.
    """
    field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_reader_one_openning_per_file(variables,data_dir)
    Reduced={}
    full_tensor=False
    for v in variables:
        try:
            full_tensor=np.stack(full_tensor,field[v])
        except:
            full_tensor=field[v]
    tensor_approx=STHOSVD(full_tensor,epsilon=tol,rank=rank)

    return full_tensor, tensor_approx,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights

def Analize_compressed_bp_vars_as_dim(bp_compressed_out, show_plot=True, plot_name="",variables=[]):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed"""
    from analysis.plot import benchmark_plotter
    full_tensor, tensor_approx,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_compressed_out
    approx_data={}
    if show_plot or (plot_name!=""):
        approx_data["SHO_SVD"]=np.stack(tucker_error_data(tensor_approx,full_tensor))
        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,
                            title="Wave simulation ST-HOPOD decomposition")

    data={}
    for i in range(full_tensor.shape[0]):
        var=variables[i]
        buff=tensor_approx.reconstruction()
        print(shape(buff))
        data[var]=

    bp_comp_to_vtk(data,X,Y,time_list,heights,
                   full_fields=field,base_name="notus_bp_comp")


def prepare_compressed_var_as_dim(tensor_approx,variables):
    """Recontruct compressed field to ndarray contained in dictionnary"""
    array_dict={}
    for var in FT_dict:
        print("field "+str(var)+" has a decomposition rank of "+str(FT_dict[var].core.shape))
        array_dict[var]=np.copy(FT_dict[var].reconstruction(),order='F')
    return array_dict


if __name__=='__main__':
    from analysis.plot import benchmark_plotter

    var_list=['density','pressure','vorticity','velocity_u','velocity_v']
    # var_list=var_list[0:2]
    data_dir="../data_notus_wave_small/"
    out_dir='../output/compressed_notus_wave/'
    # base_name="Lucas_dL010_"

    bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-6,rank=-1 )
    Analize_compressed_bp(bp_compressed_out,show_plot=False, plot_name=out_dir+"ST_HOSVD_wave_plot.pdf")

    bp_compressor_variables_as_dim(var_list,data_dir, tol=1e-6,rank=-1 )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:38:16 2018

@author: diego
"""

from interfaces.bp_reader import bp_reader,bp_reader_one_openning_per_file
from pydecomp.core.tucker_decomp import SHOPOD, STHOSVD
from pydecomp.core.Tucker import tucker_error_data, TuckerTensor
from pydecomp.core.TensorTrain import TensorTrain, error_TT_data
from pydecomp.core.TT_SVD import TT_SVD
import pydecomp.core.tensor_algebra as ta
import pydecomp.core.MassMatrices as mm
import numpy as np
from interfaces.output_vtk import VTK_save_space_time_field

from pydecomp.utils.IO import save
from pydecomp.analysis.plot import simple_1D_plot

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
    # field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_reader(variables,data_dir)
    field, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_reader_one_openning_per_file(variables,data_dir,tensorized=False)
    Reduced={}
    for v in variables:
        vfield=field[v]
        print("field shape('"+v+"') :\t", vfield.shape)
        Reduced[v]=STHOSVD(vfield,epsilon=tol,rank=rank)
    save(Reduced,'../output/compressed_notus_wave/decomp.dat')
    return field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights


def Analize_compressed_bp(bp_compressed_out, show_plot=True, plot_name=""):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed"""
    from pydecomp.analysis.plot import exp_data_decomp_plotter
    field, Reduced,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_compressed_out
    approx_data={}
    if show_plot or (plot_name!=""):
        for var in Reduced:
            tucker_decomp=Reduced[var]
            print(var,"decomp rank",tucker_decomp.core.shape)
            F=field[var]
            simple_1D_plot(tucker_decomp.u[0][:,:3], np.linspace(9,11,3),
                        x_label="Wave height", show=False,
                        plot_name="../output/compressed_notus_wave/"+var+"_height_modes_plot.pdf")
            simple_1D_plot(tucker_decomp.u[1][:,:5], np.linspace(0,10000,201),
                        x_label="time", show=False,
                        plot_name="../output/compressed_notus_wave/"+var+"_time_modes_plot.pdf")
            simple_1D_plot(tucker_decomp.u[3][:,:5], np.linspace(0,0.6,256),
                        x_label="X", show=False,
                        plot_name="../output/compressed_notus_wave/"+var+"_X_modes_plot.pdf")
            simple_1D_plot(tucker_decomp.u[2][:,:5], np.linspace(0,0.6,256),
                        x_label="Y", show=False,
                        plot_name="../output/compressed_notus_wave/"+var+"_Y_modes_plot.pdf")
            # approx_data[var]=np.stack(tucker_error_data( tucker_decomp,F))
            # exp_data_decomp_plotter(approx_data, show_plot, plot_name=plot_name,
            #                   title="Wave simulation ST-HOSVD decomposition")
            new_shape=(F.shape[0],F.shape[1],-1)
            field[var]=F.reshape(new_shape)
            Reduced[var]=np.reshape(tucker_decomp.reconstruction(),new_shape)
    # bp_comp_to_vtk(Reduced,X,Y,time_list,heights,
    #                full_fields=field,base_name="notus_bp_comp")


def bp_comp_to_vtk(fields_approx,X,Y,time_list, param_list,
                   full_fields=None,base_name="notus_bp_comp"):
    """
    Wrapper for bp decomposed fields saving to vtk format
    It is assumed that tucker_fields is a dictionnary of tucker format fields
    """
    print(type(param_list))
    out_dir="../output/compressed_notus_wave/"
    if type(list(fields_approx.values())[0])==TuckerTensor:
        var_dic=FT_dict_to_array(fields_approx)
    else:
        var_dic=fields_approx

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
        if FT_dict[var].ndim==3:
            array_dict[var]=np.copy(FT_dict[var].reconstruction(),order='F')
        elif FT_dict[var].ndim==4:
            shape=FT_dict[var].shape
            array_dict[var]=np.reshape(FT_dict[var].reconstruction(),
                                       (shape[0],shape[1],-1),order='F')
        else:
            raise NotImplementedError("Bad number of dimension")
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
    full_tensor, nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_reader_one_openning_per_file(variables,data_dir)
    tensor_approx={}
    tensor_approx["SHO_SVD"]=STHOSVD(full_tensor,epsilon=tol,rank=rank)
    tensor_approx["TT_SVD"]=TT_SVD(full_tensor,eps=tol,rank=rank)

    return full_tensor, tensor_approx,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights

def Analize_compressed_bp_vars_as_dim(bp_compressed_out, show_plot=True, plot_name="",variables=[]):
    """ Encapsulates the analysis to ease the reading. Also provides write function.
    *bp_compressed_out* is the full output of bp compressed"""
    from pydecomp.analysis.plot import benchmark_plotter
    full_tensor, tensor_approx,nxC,nyC,nx_glob,ny_glob, X,Y,time_list,heights=bp_compressed_out
    #Error plot
    approx_data={}
    if show_plot or (plot_name!=""):
        print("{} Rank ST-HOSVD decomposition with tol 1e-2".format(tensor_approx["SHO_SVD"].core.shape))
        approx_data["SHO_SVD"]=np.stack(tucker_error_data(tensor_approx["SHO_SVD"],full_tensor))
        print("{} Rank TT decomposition with tol 1e-2".format(tensor_approx["TT_SVD"].rank))
        approx_data["TT_SVD"]=np.stack(error_TT_data(tensor_approx["TT_SVD"],full_tensor))

        benchmark_plotter(approx_data, show_plot, plot_name=plot_name,
                            title="Wave simulation decomposition")

    #Export to vtk
    tensor_approx=tensor_approx["SHO_SVD"]
    data={}
    field={}
    buff=tensor_approx.reconstruction()
    for i in range(full_tensor.shape[0]):
        var=variables[i]
        data[var]=np.reshape(buff[i],(full_tensor.shape[1],full_tensor.shape[2],-1))
        field[var]=np.reshape(full_tensor[i],(full_tensor.shape[1],full_tensor.shape[2],-1))
    bp_comp_to_vtk(data,X,Y,time_list,heights,
                   full_fields=field,base_name="notus_bp_comp_VAR_AS_DIM_")




if __name__=='__main__':
    from pydecomp.analysis.plot import benchmark_plotter

    var_list=['density','pressure','vorticity','velocity_u','velocity_v']
    # var_list=var_list[0:2]
    data_dir="../data_notus_wave/"
    out_dir='../output/compressed_notus_wave/'
    # base_name="Lucas_dL010_"

    bp_compressed_out=bp_compressor(var_list,data_dir, tol=1e-6,rank=-1 )
    Analize_compressed_bp(bp_compressed_out,show_plot=False, plot_name=out_dir+"ST_HOSVD_wave_plot.pdf")

    bp_compressor_variables_as_dim(var_list,data_dir, tol=1e-6,rank=-1 )

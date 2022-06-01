#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  21 18:34:31 2018

@author: Lucas

This file encompasse the necessary work for multidimensional interpolation ROM.
"""
from interp_tensor_field import interpolate_modes
from pydecomp.core.Tucker import TuckerTensor
from interfaces.tapan_LDC_decomp import *
from pydecomp.utils.IO import load,save
from pydecomp.core.tensor_algebra import norm

import numpy as np

def interp_modes_ROM(T_approx,X,x_t,dim,method="linear"):
    """
    This function builds ROM that evaluates slices at given parameter value
    x_t that's within the original grid X. It is done by interpolating at each
    retained dim mode T_approx at x_t.

    return:
    ======
        A reconstruction of the slice of T_approx interpolated at x_t, i.e. a tensor
        of dimension d-1.
    """

    if type(T_approx)==TuckerTensor:
        U_dim=T_approx.u[dim]
        if U_dim[:,0].size != X.size:
            raise AttributeError("Shapes do not match {} \= {}".format(T_approx.u[:,0].size, X.size))
        U_interp=interpolate_modes(U_dim,X,x_t,method)
        print(U_interp)
        New_U=T_approx.u
        New_U[dim]=np.expand_dims(U_interp,axis=0)
        T_interp=TuckerTensor(T_approx.core,New_U)
        return T_interp.reconstruction()
    else:
        raise NotImplementedError(type(T_approx)+"is not available yet")

if __name__=='__main__':
    Re_list=[10000,10020,10060,10080,10100]
    path='/home/lestandi/Documents/data_ldc/grid_257x257/'
    layouts=["vectorized"]
    # LDC_multi_Re_decomp(path,Re_list,layouts,tol=1e-4,show_plot=True,
    #                     plot_name="../output/LDC_compr_data/decomp_error_graph.pdf")

    data_file='LDC_binary_Re_{}.dat'.format(Re_list)
    decomp_path="../output/LDC_compr_data/compressions_dict.dat"
    plot_path="../output/LDC_interp_data/"
    data_path=path+data_file

    T_full=load(path+'LDC_binary_Re_{}.dat'.format([10000,10020,10040,10060,10080,10100]))

    X=np.asarray(Re_list)
    Re_target=10040
    T_approx=load(decomp_path)
    T_HOSVD=T_approx["SHO_SVD vectorized"]
    T_interp=interp_modes_ROM(T_HOSVD,X,Re_target,dim=2,method="cubic")
    print(T_interp.shape)
    plot_vorticity_exponential_contour(T_interp[:,6:10,0],path=plot_path,
                                       output_name='reconstructed_vorticity_contour_plot.pdf'
                                       ,n_contour=21,centered=False,t='1900')
    plot_vorticity_exponential_contour(T_full[:,6:10,2],path=plot_path,
                                       output_name='Original_vorticity_contour_plot.pdf'
                                       ,n_contour=21,centered=False,t='1900')

    diff= np.reshape(T_interp[:,6,0]-T_full[:,6,2],(1,257*257))
    print("Rec error : {}".format( norm(T_interp[:,:,0]-T_full[:,:,2])/norm(T_full[:,:,2])))
    plot_spatial_modes(diff,path=plot_path,
                        output_name='diff_vorticity_contour_plot.pdf',
                        max_contour=3,min_contour=-3,n_contour=31)

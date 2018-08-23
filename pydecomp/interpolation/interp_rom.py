#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  21 18:34:31 2018

@author: Lucas

This file encompasse the necessary work for multidimensional interpolation ROM.
"""
from interp_tensor_field import interpolate_modes
from core.Tucker import TuckerTensor

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
        New_U=T_approx.u
        New_U[dim]=U_interp
        T_interp=TuckerTensor(T_approx.core,New_U)
        return T_interp.reconstruction()
    else:
        raise NotImplementedError(type(T_approx)+"is not available yet")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  21 18:34:31 2018

@author: Lucas

This file encompasse the necessary work for multidimensional interpolation.
"""
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import lagrange, interp1d
import time

def interpolate_modes(phi,xi,x_t,method):
    """
        This function returns the lagrange interpolation of each column (mode) of X at s_target.
        typically phi_i is known at xi points and we want its value at x_t with
        given method

        Parameters :
            phi       A 2D array (m,r) where columns are different modes
                      and with values at points xi
            xi        A set of coordinates (1D array (m,))
            x_t A target coordinate

        Return:
            phi_t     A 1D array (N,) containing modes interpolation at target coordinate
    """
    r=phi.shape[1]
    phi_t=[]
    for i in range(r):
        phi_i_interp= interpolate_with_method(phi[:,i],xi,x_t,method)
        phi_t.append(phi_i_interp)

    return np.asarray(phi_t)

def interpolate_with_method(phi,xi,x_t,method):
    """
        This function returns the lagrange interpolation of each column (mode) of X at s_target.
        typically phi_i is known at xi points and we want its value at x_t with
        given method

        Parameters :
            phi       A 1D array (m) with values at points xi
            xi        A set of coordinates (1D array (m,))
            x_t       A target coordinate

        Return:
            phi_t     Interpolated value at target coordinate
    """
    interp1d_methods=['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', 'next']
    if method=='lagrange':
        phi_t= lagrange(xi,phi)(x_t)
    elif method in interp1d_methods:
        phi_t= interp1d(xi,phi,method)(x_t)
    else:
        raise NotImplementedError(method+" is not an available interpolation method")

    return phi_t

def interp_field(Fi, Si, s_target):
    """
        This function returns the lagrange interpolation of each line of Fi at s_target.
        It wont be efficient for large arrays, then a fortran routine performing the same should be coded.

        Parameters :
            Fi        A 2D array (N,m) where colums are values at points Si and lines contains vectors to be interpolated
            Si        A set of coordinates (1D array (m,))
            s_target  A target coordinate

        Return:
            F_target  A 1D array (N,)
    """
    # Fi = np.asanyarray(Fi).transpose()
    print(np.shape(Fi))
    print('size of interpolation along X:',  np.size(Fi,0))

    # t2= time.perf_counter()
    lagr=poly_lagrange_at_target(Si,s_target)
    # lagr=np.ones(np.size(Fi,1))/(np.size(Fi,1)) #simple averaging
    F_target = np.matmul(Fi,lagr)


    return F_target


def poly_lagrange_at_target(X, x_bar):
    """
        Returns the langrange polynomials computed at interpolation point
        Parameters:
            X    An array of coordinates
            x_bar  target coordinate
        Return:
            l_i    the values of the lagrange polynomials at x_bar
    """
    n=X.size
    l_i = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j: l_i[i]*=(x_bar-X[j])/(X[i]-X[j])

    return l_i

if __name__=="__main__":
    nx=5
    ny=2
    X=np.linspace(0,1,nx)
    print(X)
    x_t=0.35
    def func(x):
        return pow(x,3) #  np.exp(x)+np.log(x)
    def f2(x):
        return pow(x,1)+1
    print(str(func(x_t)))
    F=np.zeros((nx,ny))
    F[:,0]=func(X)
    F[:,1]=f2(X)
    print(F)
    f_t=interpolate_modes(F,X,x_t,method="spline")
    print("f(x_t={})={}".format(x_t,f_t))
    print('error'+str(func(x_t)-f_t[0]))
    print('error'+str(f2(x_t)-f_t[1]))

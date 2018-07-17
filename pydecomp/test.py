# -*- coding: utf-8 -*-
"""
Created on 28/06/2018

@author: Lucas Lestandi

Simple test procedure for pydecomp library
"""
from analysis.benchmark_multivariable import benchmark_multivariable

decomp_methods=["RPOD","SHO_POD","TT_SVD"]#,"PGD"]"HO_POD",
solver=["trapezes","trapezes","SVD"]#,'trapezes']"trapezes",
# decomp_methods=["TT_SVD"]#,"PGD",
# solver=["SVD"]#]'trapezes',
# shape=[30 for i in range(3)]
shape=[20,100,100,100]
benchmark_multivariable(decomp_methods, solver ,shape=shape[::-1],
                            test_function=2, plot=True,output_decomp='',
                             plot_name='',tol=1e-8)
                            #plot_name='',tol=1e-6)

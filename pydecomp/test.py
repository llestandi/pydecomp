# -*- coding: utf-8 -*-
"""
Created on 28/06/2018

@author: Lucas Lestandi

Simple test procedure for pydecomp library
"""
from analysis.benchmark_multivariable import benchmark_multivariable

decomp_methods=["RPOD","HO_POD","SHO_POD","TT_SVD"]#,"PGD"]
solver=["trapezes","trapezes","trapezes","SVD"]#,'trapezes']
# decomp_methods=["TT_SVD"]#,"PGD",
# solver=["SVD"]#]'trapezes',
<<<<<<< HEAD
benchmark_multivariable(decomp_methods, solver ,shape=[20,15,20],
=======
benchmark_multivariable(decomp_methods, solver ,shape=[30 for i in range(5)],
>>>>>>> e31e004bfd3ec13e83d2a7fa19a79334ba2c5064
                            test_function=2, plot=True,output_decomp='',
                             plot_name='output/approx_benchmark.pdf',tol=1e-8)
                            #plot_name='',tol=1e-6)

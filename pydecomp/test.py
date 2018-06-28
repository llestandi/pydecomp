# -*- coding: utf-8 -*-
"""
Created on 28/06/2018

@author: Lucas Lestandi

Simple test procedure for pydecomp library
"""
from analysis.benchmark_multivariable import benchmark_multivariable

decomp_methods=["RPOD","HO_POD","SHO_POD","PGD"]#,"TT_SVD"]#,
solver=["trapezes","trapezes","trapezes",'trapezes']#,"SVD"]#]
benchmark_multivariable(decomp_methods, solver ,shape=[20,20,20],
                            test_function=2, plot=True,output_decomp='',
                            plot_name='output/approx_benchmark.pdf',tol=1e-6)

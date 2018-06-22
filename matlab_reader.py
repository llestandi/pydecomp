#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:17:31 2018

@author: diego
"""

import numpy as np
import h5py
from SHOPOD import SHOPOD
from high_order_decomposition_method_functions import unit_mass_matrix_creator
from plot_error_tucker import plot_error_tucker
from plot_error_tt import plot_error_tt
from TT_SVD2 import TT_SVD2


def matlab_file_reduction(file_name, error_evaluation=False):
    '''
    Etapes pour lire un fichier de matlab V 7.3
    '''

    f=h5py.File(file_name,'r')
    data=f[list(f.keys())[0]]
    data=np.array(data)
    # Mass_matrix=unit_mass_matrix_creator(data)
    # U=SHOPOD(data,Mass_matrix)
    U=TT_SVD2 (data)

    if error_evaluation:
        # plot_error_tucker(U,data,1, 'Error vs compression rate',
        #                   output_variable_name='Matlab_compression_file')
        plot_error_tt(U,data,1, 'Error vs compression rate',
                          output_variable_name='Matlab_compression_file')
    return U

if __name__=='__main__':
    matlab_file_reduction("exp_data/Exemple_1.mat",error_evaluation=True)

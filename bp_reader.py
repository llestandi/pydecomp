#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:41:36 2018

@author: diego
"""

import numpy as np
from glob import glob
import adios as ad

def bp_reader(Variable,datadir):
    """
    This functions serts to read data from a bp folder for a specific
    variable.
    """
    #reading all files in folder
    files_list=glob(datadir+'*.bp')
    #Taking only the names
    files_list2=[x.split('/')[-1] for x in files_list]
    #Detecting the first criteria
    first_criteria=[]
    for i in range(len(files_list2)):
        splited_variable=files_list2[i].split('_')
        if splited_variable[1] not in first_criteria:
            first_criteria.append(splited_variable[1])
    #files will contain all the files of the folder
    #files=[]
    first_criteria=sorted(first_criteria)
    print(first_criteria)
    '''
    for i in range(len(files_list)):
        f=ad.File(files_list[i])
        files.append(f)
    '''
    #Here each file will go to its family according first criteria, so Tensor
    #is a list of list with the names of the files organized according to
    #first criteria.
    Tensor=[]
    for i in range(len(first_criteria)):
        Tensor.append([])

    """
    for i in range(len(files_list2)):
        for j in range(len(first_criteria)):
            if files_list2[i].split('_')[1]==first_criteria[j]:
               if Tensor[j]==[]:
                   f=ad.File(files_list[i])
                   Tensor[j]=f[Variable].read()
                   a=Tensor[j].shape[0]
                   b=Tensor[j].shape[1]
                   Tensor[j]=Tensor[j].reshape([1,(a*b)])
               else:
                   f=ad.File(files_list[i])
                   aux=f[Variable].read()
                   a=aux.shape[0]
                   b=aux.shape[1]
                   aux=aux.reshape([1,(a*b)])
                   Tensor[j]=np.append(Tensor[j],aux,axis=0)
    """
    #he we clasify each file name by their first criteria
    for i in range(len(files_list2)):
        for j in range(len(first_criteria)):
           if files_list2[i].split('_')[1]==first_criteria[j]:
               if Tensor[j]==[]:
                   Tensor[j]=[files_list2[i]]
               else:
                   Tensor[j].append(files_list2[i])

    #Looking the time step we re arange in increasing order
    for i in range(len(Tensor)):
        Tensor[i]=sorted(Tensor[i])

    #We replace each element of Tensor for the variable contained in
    #the file with the same name.
    FinalTensor=[]
    a=0
    b=0
    nx_global=0
    ny_global=0
    X=0
    Y=0


    for i in range(len(Tensor)):
        FinalTensor.append([])
        for j in range(len(Tensor[i])):
            print(i,j,Tensor[i][j])
            if j==0:
                f=ad.File(datadir+Tensor[i][j])
                aux=f[Variable].read()
                a=aux.shape[0]
                b=aux.shape[1]
                aux=aux.reshape([1,(a*b)])
                FinalTensor[i]=aux
                if i==0:
                    nx_global=f['nx_global'].read()
                    ny_global=f['ny_global'].read()
                    X=f['X'].read()
                    Y=f['Y'].read()

            else:
                f=ad.File(datadir+Tensor[i][j])
                aux=f[Variable].read()
                a=aux.shape[0]
                b=aux.shape[1]
                aux=aux.reshape([1,(a*b)])
                FinalTensor[i]=np.append(FinalTensor[i],aux,axis=0)


    return   FinalTensor, a, b, nx_global, ny_global, X, Y

if __name__=='__main__':
    datadir='data_notus_wave/'
    bp_reader('density',datadir)

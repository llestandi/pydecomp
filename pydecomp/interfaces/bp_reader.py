#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:41:36 2018

@author: diego
"""

import numpy as np
from glob import glob
import adios as ad

def bp_reader(Variables,datadir):
    """
    This functions serts to read data from a bp folder for a specific
    variable.
    """
    name_Tensor,first_criteria=parse_notus_files(datadir)
    #We replace each element of name_Tensor for the variable contained in
    #the file with the same name.
    f=ad.File(datadir+name_Tensor[0][0])
    nx_global=f['nx_global'].read()
    ny_global=f['ny_global'].read()
    X=f['X'].read()
    Y=f['Y'].read()
    Tensor_dict={}
    for var in Variables:
        try:
            FinalTensor=[]
            time_list=[] #storing time stamps
            for i in range(len(name_Tensor)):
                FinalTensor.append([])
                for j in range(len(name_Tensor[i])):
                    f=ad.File(datadir+name_Tensor[i][j])
                    aux=f[var].read()
                    nxC=aux.shape[0]
                    nyC=aux.shape[1]
                    aux=aux.reshape([1,(nyC*nxC)])
                    if i==0:
                        time_list.append(name_Tensor[i][j][-9:-3])
                    if j==0:
                        FinalTensor[i]=aux
                    else:
                        FinalTensor[i]=np.append(FinalTensor[i],aux,axis=0)
            for t in FinalTensor:
                if t.shape != FinalTensor[0].shape:
                    raise AttributeError('bp Shapes differ',t.shape,FinalTensor[0].shape)
            FinalTensor=np.stack(FinalTensor,axis=0)
            Tensor_dict[var]=FinalTensor
        except:
            raise Exception('bp_reader was enable to find Variable:'+var)
        print(".bp reading successful")
    return   Tensor_dict, nxC, nyC, nx_global, ny_global, X, Y,time_list,first_criteria

def bp_reader_one_openning_per_file(Variables,datadir):
    """
    This functions serts to read data from a bp folder for a specific
    variable.
    """
    name_Tensor,first_criteria=parse_notus_files(datadir)
    print(name_Tensor)
    #We replace each element of name_Tensor for the variable contained in
    #the file with the same name.
    f=ad.File(datadir+name_Tensor[0][0])
    nx_global=f['nx_global'].read()
    ny_global=f['ny_global'].read()
    X=f['X'].read()
    Y=f['Y'].read()
    Tensor_dict={}
    for var in Variables:
        Tensor_dict[var]=[]
    time_list=[] #storing time stamps
    for i in range(len(name_Tensor)):
        for var in Variables:
            Tensor_dict[var].append([])
        for j in range(len(name_Tensor[i])):
            f=ad.File(datadir+name_Tensor[i][j])
            for var in Variables:
                aux=f[var].read()
                nxC=aux.shape[0]
                nyC=aux.shape[1]
                # aux=aux.reshape([1,(nyC*nxC)])
                if i==0:
                    time_list.append(name_Tensor[i][j][-9:-3])
                if j==0:
                    Tensor_dict[var][i]=[aux]
                else:
                    Tensor_dict[var][i].append(aux)

    for var in Variables:
        Tensor_dict[var]=np.asarray(Tensor_dict[var])

    full_tensor=np.stack([Tensor_dict[var] for var in Variables])
    print("global shape",full_tensor.shape)
    print("quick open .bp reading successful")
    return   Tensor_dict, nxC, nyC, nx_global, ny_global, X, Y,time_list,first_criteria

def parse_notus_files(path,ext=".bp"):
    """ This function parses the list of .bp files in path and Returns
    an ordered list (with 2 parameters)"""
    #reading all files in folder
    files_list=glob(path+'*'+ext)
    #Taking only the names
    files_list2=[x.split('/')[-1] for x in files_list]
    #Detecting the first criteria
    first_criteria=[]
    for i in range(len(files_list2)):
        splited_variable=files_list2[i].split('_')
        if splited_variable[1] not in first_criteria:
            first_criteria.append(splited_variable[1])
    #files will contain all the files of the folder
    first_criteria=sorted(first_criteria)
    #Here each file will go to its family according first criteria, so name_Tensor
    #is a list of list with the names of the files organized according to
    #first criteria.
    name_Tensor=[]
    for i in range(len(first_criteria)):
        name_Tensor.append([])
    #he we clasify each file name by their first criteria
    for i in range(len(files_list2)):
        for j in range(len(first_criteria)):
           if files_list2[i].split('_')[1]==first_criteria[j]:
               if name_Tensor[j]==[]:
                   name_Tensor[j]=[files_list2[i]]
               else:
                   name_Tensor[j].append(files_list2[i])

    #Looking the time step we re arange in increasing order
    for i in range(len(name_Tensor)):
        name_Tensor[i]=sorted(name_Tensor[i])

    return name_Tensor, first_criteria
if __name__=='__main__':
    datadir='data_notus_wave/'
    FinalTensor, nxC, nyC, nx_global, ny_global, X, Y,time_list=bp_reader('density',datadir)

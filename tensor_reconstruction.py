# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:11:27 2018

@author: Diego Britez
"""
import numpy as np
import readfile
from new_iteration_result import new_iteration_result 
"""
This code serve to obtain the tensor from the modes created with PGD function
"""



def reconstruction():
    tshape=readfile.read("tshape.csv")
    tshape=tshape.astype(int)
    dim=tshape.size
    U=[]
    for i in range(dim):
        aux=readfile.read("U("+str(i)+").csv")
        U.append(aux)
    aux2=U[0].shape
    rank=aux2[0]

    Resultat=np.zeros(shape=tshape)

    R=[]
    for i in range(rank):
        for j in range(dim):
            r=U[j][i][:]
            R.append(r)
        Resultat=new_iteration_result(R, tshape, Resultat)
        R=[]
        
    return Resultat
help(reconstruction)
reconstruction()
"this is a test"





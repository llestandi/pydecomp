# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:54:58 2018

@author: Diego Britez
"""

import numpy as np
import csv
from orthogonality_verification import orthogonality_verification

def Orthogonal(vector,Xgrid, dim, rank):
    
    
    Ortho_Verif=[]
    for i in range(dim):
        Ortho_Verif.append(np.zeros(rank*rank).reshape(rank,rank))

    for i in range (dim):
       for j in range(rank):
          Ortho_Verif[i][j,:]=orthogonality_verification(Xgrid[i],vector[i][j,:],vector[i])
         
    for i in range(dim): 
        nom_aux="Orthogonal_Verification_Mode_["+str(i)+"].csv"
        with open(nom_aux,'w',newline='') as fp:
            a=csv.writer(fp, delimiter=',')
            a.writerows(Ortho_Verif[i])
            nom_aux2='readfile.read("'+nom_aux+'")'
            print('--------------------------------------------------------------\n')
            print('The file', nom_aux , 'has been created,to read it, type:\n',
                  nom_aux2)

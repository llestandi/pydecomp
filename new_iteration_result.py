# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:17:40 2018

@author: Diego Britez
"""

import numpy as np

def new_iteration_result(R, tshape, Resultat):
     
     NewModeResultat=(np.transpose(np.array([R[0]])).dot(np.array([R[1]])))
     
     if len(R)>2:
         for i in range(2,len(R)):
             NewModeResultat=np.kron(NewModeResultat,R[i])
             
         NewModeResultat=NewModeResultat.reshape(tshape)
     
     Resultat=np.add(Resultat,NewModeResultat)
        
        
        
     return Resultat 
 


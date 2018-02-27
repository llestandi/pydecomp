# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:19:52 2018

@author: Diego Britez
"""
import numpy as np
def orthogonality_verification(w,R,RR):
    aux=np.transpose(np.multiply(w,RR))
    Orth=np.dot(R,aux)
    return Orth
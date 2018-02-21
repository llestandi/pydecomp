# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:17:40 2018

@author: Diego Britez
"""

import numpy as np

def new_iteration_result(R,S):
     Fres=np.transpose(S).dot(R)
     return Fres 
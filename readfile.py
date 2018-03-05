# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:30:04 2018

@author: Diego Britez
"""
import numpy as np

def readSS():
     return np.loadtxt("SS.csv",delimiter=",",dtype=float)

def readRR():
     return np.loadtxt("RR.csv",delimiter=",",dtype=float)

def readOrthox():
    return np.loadtxt("Orthox.csv",delimiter=",",dtype=float)

def readOrthoy():
    return np.loadtxt("Orthoy.csv",delimiter=",",dtype=float)






#np.loadtxt("RR.csv",delimiter=",",dtype=float)
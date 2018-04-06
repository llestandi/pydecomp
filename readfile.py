
"""
Created on Mon Mar  5 09:30:04 2018

@author: Diego Britez
"""
import numpy as np
import csv
"""
def read(file):
    with open(file, newline=""):
        lecture=csv.reader(file)
        
    return lecture        
"""        


def read(file):
    
    return np.loadtxt(file,delimiter=",",dtype=float)
    



if __name__=="__main__":
    p=read('U(1).csv')



#np.loadtxt("RR.csv",delimiter=",",dtype=float)
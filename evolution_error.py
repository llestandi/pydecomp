# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:45:46 2018

@author: Diego Britez
"""

import numpy as np
import matplotlib.pyplot as plt
class evolution_error():
    def __init__(self):
        self.R=[]
        self.error=[]
    def _new_loop_evaluation(self,r,ERROR):
        self.R.append(r)
        self.error.append(ERROR)
    def _plot_error(self):
        
        
        plt.ylim(1e-17, 0.5)
        plt.xlim(0,21)
        plt.plot(self.R,self.error,color='b', marker='o',linestyle='-.',
                 label='relative error f5')
        plt.yscale('log')
        plt.xlabel("Enrichment's mode")
        plt.ylabel('Relative Error')
        plt.grid()
        plt.show()

        
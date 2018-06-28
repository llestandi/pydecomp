# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:37:24 2018

@author: Diego Britez
"""

import numpy as np
"""
In this class the vectors who defin R^n are defined and created in to a list
called (inside the class) X.
dim --> dimention of the space
div_tshape --> array with the number of elements in each space
lower_limit --> is the lower limit value of each space
upper_limit --> is the upper limit value of each space

"""

class CartesianGrid:
    # @Diego Name does not reflect what is going on here. First you dont need a class to do that.
    # A simple function can do the job. This calss actually describes the domain (which may be
    # interesting). Then several creator could be used and it makes sense, for instance a cartesian
    # grid either equispaced or not.

    def __init__(self, dim, tshape, lower_limit, upper_limit):

        if (dim != np.size(tshape)):
            print('####### ERROR ####')
            print('Declared dimention is not congruent with div_tshape \
                  number of elements, they have to be equals')
        if (dim != np.size(lower_limit)):
            print('####### ERROR ####')
            print('Declared dimention is not congruent with lower_limit \
                  number of elements, they have to be equals')
        if (dim != np.size(upper_limit)):
            print('####### ERROR ####')
            print('Declared dimention is not congruent with upper_limit \
                  number of elements, they have to be equals')

        self.dim = dim
        self.tshape = tshape
        self.X = []
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def SpaceCreator(self):

        for i in range(self.dim):
            self.X.append(np.linspace(self.lower_limit[i],self.upper_limit[i],
                                      self.tshape[i]))

        return self.X

if __name__=="__main__":
    lower_limit = np.array([0,0])
    upper_limit = np.array([1,1])
    tshape=np.array([5,6])
    dim=2
    Vector=CartesianGrid(dim,tshape, lower_limit, upper_limit)
    Vector.SpaceCreator()

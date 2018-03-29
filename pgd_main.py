# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018

@author: Diego Britez
"""

import numpy as np
from fixepoint import fixpoint
from scipy.linalg import norm
import matplotlib.pyplot as plt
from matrixf import matfunc
from new_iteration_result import new_iteration_result
from orthogonality_verification import orthogonality_verification
from Orthogonal import Orthogonal
from IPython import get_ipython
from Canonical import CanonicalForme
from evolution_error import evolution_error
from tensor3d import tensorfunc3
from tensor4d import tensorfunc4
import CartesianGrid
import csv
import integrationgrid






 
#Definition of variables n dimention
dim=2
#------------------------------------------------------------------------------
#Definition of the number of divitions of each space 
tshape=np.zeros(dim,dtype=int)


#definition de la quantité d'élements dans le domaine x 
tshape[0]=5               
#definition de la quantité d'élements dans le domaine y
tshape[1]=6

#tshape[2]=50

#tshape[3]=60

#If the study needs more dimetions have to be added. Ex: div_tshape[2]=55  

#------------------------------------------------------------------------------
#Declaration of the limits of integration

#a et b sont les limites dans le domaine x                      
a=0                    
b=1
#c et d sont les limites dans le domaine y 
c=0                               
d=1

#The limits can be specified direcly in the lower_limit and upper_limit arrays
lower_limit=np.array([0,0])
upper_limit =np.array([1,1])

#Vector is the class where all the variables that define the space and the
#vector is storage
Vector = CartesianGrid.CartesianGrid(dim, tshape, lower_limit, upper_limit)

# SpaceCreator is the method that creates the Cartesian grid to work
X = Vector.SpaceCreator()

#------------------------------------------------------------------------------
#Creation of the matrix for study

#matfunc is a function that will create the space of data to study from
#a formule, de space defined by X. This function has to be changed for each
#problem 

F=matfunc(X, tshape)

#If the function is in 3d the file tensorfunc will help to generate
#the tensor from a equeation, the next line should be left as a 
#commentary if not working in a 3d approach

#F=tensorfunc3(X, tshape)

#If the function is in 4d the file tensorfunc will help to generate
#the tensor from a equeation, the next line should be left as a 
#commentary if not working in a 3d approach

#F=tensorfunc4(X, tshape)





#Definition de la matrice d'étude a partir des valeures
#aléatoires, il faut marquer comme commentaire si c'est
#l'option à utiliser

         
#F=np.random.randn(tshape[0], tshape[1])  

  
  
"""
This code use the PGD method to create the modes decoposition of a field of
data in n dimentions, creating n files, one file for each dimention.
The description of the PGD method as well each variable used in this code
can be found in the manuscrip of Lucas Lestandi's  doctoral thesis at
Chapter 3 (Multivariate problem decomosition).
"""                                                                         
#------------------------------------------------------------------------------
                                 

#Stop criteria value for the enrichment loop                                  
epenri=1*10**-12
                 
#Start value of epn that allows get in to the enrichment loop for the first
#iteration
epn=1                             
                               
#The variable z is created to count the number of iterations in the fixed-point
#loop which don't arrive to the convergence after maximal times of iterations
#declared as a stoping criteria.
z=0                               
                                  
#Creation d'une grille pour l'integration a partir des valeurs

Xgrid=integrationgrid.IntegrationGrid(X,dim,tshape)   
Xgrid=Xgrid.IntegrationGridCreation()
                                                             
#La variable Verification is going to be used to evaluate the orthogonality
                                     
Verification=1                   

               
    

C=CanonicalForme(tshape,dim)
evaluation=evolution_error()
C._solution_initialization()
Resultat=np.zeros(tshape)
                        

while   (epn>=epenri):
        C._set_rank()
            
            
       
        if (C._rank==1 ):
          print('Number of iterations in fixed-point routine for each enrichment')
          print('--------------------------------------------------------------\n')  
        
        R,z=fixpoint(X,C._tshape,C._U,F,z ,C._rank)
        
        #Sn=S/norm(S)
        #-------------------------------------------------------------
        for i in range(dim):
            
            Orth=orthogonality_verification(Xgrid[i],R[i],C._U[i])
            for j in range(C._get_rank()):
                if (Orth[j]>1e-10):
                    print('--------------------------------------------------------------\n')
                    print('Orthogonality is not verified at mode:', C._get_rank())
                    print('Variable of failure= U(',i,')')
                    print('Value of the scalar product of failure=', Orth[j])
                    Verification=0
                
        
        #--------------------------------------------------------------        
        
        C._add_enrich(R)                                                                       
        Resultat=new_iteration_result(R,tshape,Resultat)
        
        if C._get_rank()==1:
            REF=R[dim-1]
           
        epn=norm(R[dim-1])/norm(REF)
        
        
        ERROR=norm(F-Resultat)/norm(F)
        """
        This routine serve to evaluate the evolution of the value of 
        error for each enrichment loop, if the evaluation is not desired,
        leave this section as a comentary
        """
        
        evaluation._new_loop_evaluation(C._rank,ERROR)
        
        
        if (((norm(F-Resultat)/norm(F)))<(1e-12)):
            print('--------------------------------------------------------------\n') 
            if (Verification==1):   
                print('Orthogonality between modes was verified') 
                print('----------------------------------------\n') 
            print('Max. Relative error criteria was verified')
            print('Relative error=',(norm(F-Resultat)/norm(F)),'< 1e-10' )
            break
        

    
print('--------------------------------------------------------------\n')      
print("Iteration's enrichment loops=",C._get_rank()) 
print("Iteration's enrichment loops in fixed-point loop with no convergence:",z)       
print('Epsilon=',epn)



##--------------------------------------------------------------------------
#For the simplicity in the code we name p to the rank number (number of modes)
p=C._get_rank()

#_final_result method is called to remove the first mode of zeros created
#at the begining to initiate the enrichment process.
C._final_result()


"""
This module will return the matrices of the scalar product between the         
vectors that form the RR space and SS space. This is made in order to 
prove the orthogonality between this vectors. If this module is not 
desired, leave as commentary
"""

#Orthogonal(C._U,Xgrid,C._dim,p)  


"""
Creation of the files with each mode of the solution in independent files
calling a mode in CanonicalForme
"""

C._writeU() 

"""
With the folowing command  all variables are going to be erased and the 
solution modes could have have acces by the files that were created
"""   
evaluation._plot_error()
get_ipython().magic('reset -sf') 

 



     




    
    




      
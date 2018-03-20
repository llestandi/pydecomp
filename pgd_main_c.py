# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:35:41 2018

@author: Diego Britez
"""

import numpy as np
from fixepoint import fixpoint
from scipy.linalg import norm
from matrixf import matfunc
from new_iteration_result import new_iteration_result
from orthogonality_verification import orthogonality_verification
from Orthogonal import Orthogonal
from IPython import get_ipython
import Canonical
import CartesianGrid
import csv
import integrationgrid


 

"""This application will compute la solution of function in the form of 
a matrix with the PGD method, the programing of this application is based
 in the equations that can be founded in the  'Manuscript' of the Thesis
 of Lucas Lestandi in chapter 1"""
 
#Definition of variables
dim=2
#------------------------------------------------------------------------------
#Definition of the number of divitions of each space 
tshape=np.zeros(dim,dtype=int)


#definition de la quantité d'élements dans le domaine x 
tshape[0]=200                  
#definition de la quantité d'élements dans le domaine y
tshape[1]=200

#If the study needs more dimetions have to be added. Ex: div_tshape[2]=55  

#------------------------------------------------------------------------------
#Declaration of the limits of integration
                      
a=0                               #a et b sont les limites dans le domaine x
b=1
c=0                               #c et d sont les limites dans le domaine y 
d=1

#The limits can be specified direcly in the lower_limit and upper_limit arrays
lower_limit=np.array([a,c])
upper_limit =np.array([b,d])

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

#This loop serve to verify the uncompresed file with the final result
#Remove the to allow the this function work
"""
with open ('F.csv', 'w', newline='') as file:
    write= csv.writer(file)
    for line in F:
        write.writerow(line)
"""

#Definition de la matrice d'étude a partir des valeures
#aléatoires, il faut marquer comme commentaire si c'est
#l'option à utiliser

         
#F=A = np.random.randn(ny, nx)    
                                                                    
#-----------------------------------------------------------------------------------------
                                 
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
#bet                                      
Verification=0                    #Variable qui séra utilisée pour éva-
                                  #luer l'orthogonalitée

               
    
    
""""This is the principal loop of the application where the numerical solution 
is calculated with the PGD method;
- In the first line whe activate the iteration counter for the present iteration
- In the second line the functions S and R for the present iteration are calculated calling
to the fixpoint application
- In the third line, the results of S and R are introduced as a row to the Matrix SS and RR
 wich colects the solutions found in each iteration
 - The multiplication of R and S is carried out and aditioned to the matrix of solution 
 calling to new_iteration_result application
- The first vector S1 is soraged to be used for the stoping criteria"""

C=Canonical.CanonicalForme(tshape,dim)
C._solution_initialization()

                        

while   ((epn>=epenri))  :
        C._set_rank()
            
            
        RR=C._get_U()[0]
        SS=C._get_U()[1]
        if (C._rank==1 ):
          print('Number of iterations in fixed-point routine for each enrichment')
          print('--------------------------------------------------------------\n')  
        S,R,z=fixpoint(X[0],X[1],SS,RR,C._tshape[0],C._tshape[1],F,z,C._rank)
        Sn=S/norm(S)
        #-------------------------------------------------------------
        
        OrthR=orthogonality_verification(Xgrid[0],R,RR)
        
        
       
        #Module  pour évaluer l'orthogonalité des vecteurs qui forment
        #les différents modes
        Verification=1
        for i in range (C._get_rank()):
              if (OrthR[0][i]>1e-10):
                  print('Orthogonality is not verified at mode', C._get_rank())
                  print('Scalar product value of failure=', OrthR[0][i])
                  Verification=0
                  
        OrthS=orthogonality_verification(Xgrid[1],S,SS)
        for i in range (C._get_rank()):
            if (OrthS[0][i]>1e-10):
                print('Orthogonality is not verified at mode',C._get_rank())
                print('Scalar product value of failure=', OrthS[0][i])
                Verification=0
                
        #--------------------------------------------------------------        
        
        C._add_enrich(R,S)                                                                       
        New_Result=new_iteration_result(C._U[0],C._U[1])
        
        if C._get_rank()==1:
            S1=S
        epn=norm(S)/norm(S1)
        
        
        if (((norm(F-New_Result)/norm(F)))<(1e-10)):
            print('--------------------------------------------------------------\n') 
            if (Verification==1):   
                print('Orthogonality between modes was verified') 
                print('----------------------------------------\n') 
            print('Max. Relative error criteria was verified')
            print('Relative error=',(norm(F-New_Result)/norm(F)),'< 1e-10' )
            break
    

    
print('--------------------------------------------------------------\n')      
print('Numbers of iterations of enrichment=',C._get_rank()) 
print('Numbers of iterations in fixed-point loop with no convergence:',z)       
print('Epsilon=',epn)



##--------------------------------------------------------------------------
#For the simplicity in the code we name p to the number of iterations
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
Orthogonal(C._U,Xgrid,C._dim,p)  
"""
Creation of the files with each mode of the solution in independent files
"""
C._writeU() 

"""
With the folowing command  all variables are going to be erased and the 
solution modes could have have acces by the files that were created
"""   
get_ipython().magic('reset -sf') 

 



     




    
    




      
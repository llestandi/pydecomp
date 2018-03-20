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
from Canonical import CanonicalForme
import CartesianGrid
import csv
import integrationgrid


 

"""This application will compute la solution of function in the form of 
a matrix with the PGD method, the programing of this application is based
 in the equations that can be founded in the  'Manuscript' of the Thesis
 of Lucas Lestandi in chapter 1"""
 
#Definition of variables n dimention
dim=2
#------------------------------------------------------------------------------
#Definition of the number of divitions of each space 
tshape=np.zeros(dim,dtype=int)


#definition de la quantité d'élements dans le domaine x 
tshape[0]=200                 
#definition de la quantité d'élements dans le domaine y
tshape[1]=300

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


#Definition de la matrice d'étude a partir des valeures
#aléatoires, il faut marquer comme commentaire si c'est
#l'option à utiliser

         
#F=np.random.randn(tshape[0], tshape[1])  

  
#This loop print the uncompresed solution and it will serve to 
#verify the uncompresed file with the final result.
#Remove the to allow the this function work
"""
with open ('F.csv', 'w', newline='') as file:
    write= csv.writer(file)
    for line in F:
        write.writerow(line)
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
                
                    
        """
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
        """       
        #--------------------------------------------------------------        
        
        C._add_enrich(R)                                                                       
        Resultat=new_iteration_result(R,tshape,Resultat)
        
        if C._get_rank()==1:
            REF=R[dim-1]
           
        epn=norm(R[dim-1])/norm(REF)
               
        """
        if (((norm(F-Resultat)/norm(F)))<(1e-10)):
            print('--------------------------------------------------------------\n') 
            if (Verification==1):   
                print('Orthogonality between modes was verified') 
                print('----------------------------------------\n') 
            print('Max. Relative error criteria was verified')
            print('Relative error=',(norm(F-Resultat)/norm(F)),'< 1e-10' )
            break
        """

    
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

get_ipython().magic('reset -sf') 

 



     




    
    




      
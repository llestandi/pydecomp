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
from grille import grille
from orthogonality_verification import orthogonality_verification
import csv 

"""This application will compute la solution of function in the form of a matrix with the PGD
method, the programing of this application is based in the equations that can be founded in the 
'Manuscript' of the Thesis of Lucas Lestandi in chapter 1"""
#Definition of variables

nx=78                            #definition de la quantité d'élements dans le domaine x
ny=78                             #definition de la quantité d'élements dans le domaine y
a=0                               #a et b sont les limites dans le domaine x
b=1
c=0                               #c et d sont les limites dans le domaine y 
d=1
#-----------------------------------------------------------------------------------------
#Creation of the matrix for study
X=np.linspace(a,b,nx)             #Vecteur qui discretise le domaine X
Y=np.linspace(c,d,ny)             #Vecteur qui discretise le domaine Y
F=matfunc(X,Y)                    #Matrice de la fonction déjà discretisée a partir d'une
#F=A = np.random.randn(ny, nx)     #Definition de la matrice d'étude a partir des valeures
                                  #aléatoires, il faut marquer comme commentaire si c'est
                                  #l'option à utiliser
                                                                    
#-----------------------------------------------------------------------------------------
                                  #fonction à introduir dans le fichier matrixf 
                                  
epenri=1*10**-12                  #Definition du critère d'arrêt pour la boucle principal

epn=1                             #valeure de début pour entrer dans la boucle

p=0                               #Initialization du compteur du nombre d'iterations du point fixe

SS=np.array([np.zeros(ny)])       #Initialization de la matrice qui ajoute des valeures de S
                                  #(fonction de solution dans l'éspace Y) à la
                                  #sortie de chaque iteration du point fixe
SSnorm=np.array([np.zeros(ny)])                                 

RR=np.array([np.zeros(nx)])       #Initilization de la matrice qui ajoute des valeures de R 
                                  #(fonction de solution dans l'éspace X) à la 
                                  #sortie de chaque iteration du point fixe
                                  
U=np.zeros(np.shape(F))           #Creation de la matrice de Résultat dans l'espace solution                                  

z=0                               #Variable créé pour counter des iterations dans la boucle du
                                  #du point fixe qui n'arrivent pas a la convergence après un 
                                  #nombre maximal d'iterations
                                  
Xgrid= grille(X,nx)               #Creation d'une grille pour l'integration a partir des valeurs
                                  #du vecteur X
Ygrid= grille(Y,ny)               #Creation d'une grille pour l'integration a partir des valeurs
                                  #du vecteur Y 

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
 
                    
    

while   ((epn>=epenri))  :
        p=p+1
        S,R,z=fixpoint(X,Y,SS,RR,nx,ny,F,z)
        Sn=S/norm(S)
        OrthR=orthogonality_verification(Xgrid,R,RR)
        
        #-------------------------------------------------------------
        #Module  pour évaluer l'orthogonalité des vecteurs qui forment
        #les différents modes
        Verification=1
        for i in range (p):
              if (OrthR[0][i]>1e-10):
                  print('Orthogonality is not verified at enrichment', p)
                  print('Scalar product value of failure=', OrthR[0][i])
                  Verification=0
                  
        OrthS=orthogonality_verification(Ygrid,S,SS)
        for i in range (p):
            if (OrthS[0][i]>1e-10):
                print('Orthogonality is not verified at enrichment', p)
                print('Scalar product value of failure=', OrthS[0][i])
                Verification=0
                
        #--------------------------------------------------------------        
        SS=np.append(SS,S,axis=0)
        SSnorm=np.append(SSnorm,Sn,axis=0)
        RR=np.append(RR,R,axis=0)        
        Un=new_iteration_result(R,S)
        U=U+Un
        
                
        if p==1:
            S1=S
        epn=norm(S)/norm(S1)
        
        
        if ((norm(F-U)/norm(F))<(1e-10)):
            break
        
if (Verification==1):   
    print('Orthogonality is verified') 
       
print('Numbers of iterations of enrichment=',p) 
print('Numbers of iterations in fixed-point boucle with no convergence:',z)       
print('Epsilon=',epn)
print('Erreure relative=',(norm(F-U)/norm(F)))


##--------------------------------------------------------------------------
#Orthogonality verification 
"""
This module will return the matrices of the scalar product between the         
vectors that form the RR space and SS space. This is made in order to 
prove the orthogonality between this vectors. If this module is not 
desired, leave as commentary"""

#Verification in the RR space (X)
p=p+1
Orthox=np.zeros(p*p).reshape(p,p)
for i in range (p):
        Orthox[i][:]=orthogonality_verification(Xgrid,RR[i][:],RR)
#Verification in the SS space (X)    
Orthoy=np.zeros(p*p).reshape(p,p)
for i in range (p):
    Orthoy[i][:]=orthogonality_verification(Ygrid,SS[i][:],SS)

with open('Orthox.csv','w',newline='') as fp:
     a=csv.writer(fp, delimiter=',')
     a.writerows(Orthox)
with open('Orthoy.csv','w',newline='') as fp:
     a=csv.writer(fp, delimiter=',')
     a.writerows(Orthoy)   

"""To read the values, its necesary to activate the next line      
 np.loadtxt("Orthox.csv",delimiter=",",dtype=float)
 np.loadtxt("Orthoy.csv",delimiter=",",dtype=float) 
"""    
    
    




      
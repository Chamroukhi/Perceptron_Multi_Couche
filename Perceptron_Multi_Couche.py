#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 14:42:06 2020

@author: Chamroukhi
"""

import numpy as np 
import matplotlib.pyplot as plt

def TanH (x):
        
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def TanH_derivative(x):
    
    return (1+TanH(x))*(1-TanH(x))

#Input datasets
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])



iteration = 1000
lr = 0.1

#initialiser les poids Wij
W03=-0.5
W13=-1
W23=0.4
W04=0.3
W14=-0.4
W24=0.2
#initialiser les poids Wjk
W05=-0.4
W35=-0.3
W45=-0.1
#initialiser les biais
biais=1
biais1=1



print("=========== Les valeurs initiale de poids Wij (couche caché): ============== ",end='')
print("\n W03 =",W03,"\n W13 =",W13,"\n W23 =",W23,"\n W04 =",W04,"\n W14 =",W14,"\n W24 =",W24)
print("biais de couche caché: ",end='')
print(biais)
print("========== Les valeurs initiale de poids Wjk (couche sortie): ===============",end='')
print("\n W35 =",W35,"\n W45 =",W45,"\n W05 =",W05)
print("biais de couche de sortie: ",end='')
print(biais1)

#affichage
x=0

#entrainer l'algorithme
for i in range(iteration):
    for xi in range(len(X)):  
        #Forward Propagation
        if x<4:
            print("\n")
            print("=================== itération ",i+1,"Observation ",xi+1,"================== \n")
            print("==================== FeedForward Propagation ====================== \n")
        #calcul l'activation sur neurone 3"""
        S3= round(X[xi][0]*W13 + X[xi][1]*W23 + biais*W03,3)
        a3= round(TanH(S3),3)
        #affichage
        if x<4:
            print("S3 = X1 * W13 + X2 * W23 + biais * W03")
            print("S3 =",X[xi][0],"*",W13,"+",X[xi][1],"*",W23,"+",biais,"*",W03,"=",S3)
            print("a3 = TanH(S3) =",a3,"\n")
        
        #calcul l'activation sur neurone 4
        S4= round(X[xi][0]*W14 + X[xi][1]*W24 + biais*W04,3)
        a4=round(TanH(S4),3)
        #affichage
        if x<4:
            print("S4 = X1 * W14 + X2 * W24 + biais * W04")
            print("S4 =",X[xi][0],"*",W14,"+",X[xi][1],"*",W24,"+",biais,"*",W04,"=",S4)
            print("a4= TanH(S3) =",a4,"\n")
        
        #calcul de sortie de reseaux
        Sk= round(a3*W35 + a4*W45 + biais1*W05,3)
        sortie= round(TanH(Sk),3)
        #affichage
        if x<4:
            print("Sk = a3 * W35 + a4 * W45 + biais1 * W05")
            print("Sk =",a3,"*",W35,"+",a4,"*",W45,"+",biais1,"*",W05,"=",Sk)
            print("sortie = TanH(Sk) =",sortie)
        
            print("\n")
            print("======================= Back Propagation =========================== \n")
        
        #Backpropagation"""
        #erreur sur la couche de sortie
        erreur = round(Y[xi][0]-sortie,3)
        
        #affichage
        if x<4:
            print("1) L'erreur globale erreur = Y - sortie :")
            print("e = ",Y[xi][0],"-",sortie,"=",erreur)
            print("\n")
        
        #calcul l'erreur sur les unités de sortie
        Qk =round( erreur * TanH_derivative(Sk),3)
        #affichage
        if x<4:
            print("2) calcul l'erreur sur la couche de sortie :")
            print("Qk = erreur * TanH_derivative(Sk)")
            print("Qk =",erreur,"*",round(TanH_derivative(Sk),3),"=",Qk)
            print("\n")
        
        #erreur sur les unités cachée
        Qj3=round((W35*Qk)*TanH_derivative(S3),3)
        Qj4=round((W45*Qk)*TanH_derivative(S4),3)
        #affichage
        if x<4:
            print("3) calcul l'erreur sur la couche caché :")
            print("Qj3 = ((W35 * Qk) * TanH_derivative(S3))")
            print("Qj3 =",W35,"*",Qk,"$",round(TanH_derivative(S3),3),"=",Qj3)
            print("Qj4 = ((W45 * Qk) * TanH_derivative(S4))")
            print("Qj4 =",W45,"*",Qk,"$",round(TanH_derivative(S4),3),"=",Qj4)
            print("\n")
        
            #Ajustement de poids 
            print("4) Mise a jour des poids synaptiques :")
        #poids de couche de sortie"""
        W35= round(W35+(lr*Qk*a3),3)
        W45= round(W45+(lr*Qk*a4),3)
        W05= round(W05+(lr*Qk*biais1),3)
        #affichage
        if x<4:
            print("Wjk = Wjk + lr*Qk*aj")
            print("\n W35= ",W35,"\n W45= ",W45,"\n W05= ",W05,"\n")

        #poids de couche cachée
        W13= round(W13+(lr*Qj3*X[xi][0]),3)
        W23= round(W23+(lr*Qj3*X[xi][1]),3)
        W14= round(W14+(lr*Qj4*X[xi][0]),3)
        W24= round(W24+(lr*Qj4*X[xi][1]),3)
        
        W03= round(W03+(lr*Qj3*biais),3)
        W04= round(W03+(lr*Qj4*biais),3)
        #affichage
        if x<4:
            print("Wij = Wij + lr*Qj*ai")
            print("\n W13= ",W13,"\n W23= ",W23,"\n W14= ",W14,"\n W24= ",W24,"\n W04= ",W04,"\n W03= ",W03,"\n")
            y=input("passe a létape suivante ? :")
            if y:
                print("")
            x=x+1  
        if x>4:    
            print("itération ",i)
def prediction(x1,x2):
    S3= round((x1*W13 + x2*W23 + biais1*W03),3)
    a3= round(TanH(S3),3)
        
    #calcul l'activation sur neurone 4
    S4=round(( x1*W14 + x2*W24 + biais*W04),3)
    a4=round(TanH(S4),3)
    #calcul de sortie de reseaux
    Sk= round((a3*W35 + a4*W45 + biais1*W05),3)
    sortie= round(TanH(Sk),3)
    
    return sortie

print("=========== Le valeur finale des poids Wij (couche caché): ============== ",end='')
print("\n W03 =",W03,"\n W13 =",W13,"\n W23 =",W23,"\n W04 =",W04,"\n W14 =",W14,"\n W24 =",W24)

print("========== La valeur finale des poids Wjk (couche sortie): ===============",end='')
print("\n W35 =",W35,"\n W45 =",W45,"\n W05 =",W05)

#Prédiction
print("============================== Prédiction ================================ \n")
for x in range(4):
    print(" donner x1 et x2 :")
    X1=int(input("X1 = "))
    X2=int(input("X2 = "))
    print("\n")
    print("Y= ",prediction(X1,X2),"\n")

#! /usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 2 — Arbre de décision, Bayes Naïf et KNN

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
Dl pycharm sur linux
"""
from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import Decision Tree Classifie
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt

from color import center_color,crop_center
from fourier_transform import fourier_transform
from binaryPattern import binaryPatterns
import match 
import operator

import random
import math
########################################   Initialisations   ########################################

#image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
#image_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/images/'
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"
#dataset_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_label_data_set.csv'
dataset_path = "/home/alex/Desktop/GTI770-tp2/csv/galaxy/galaxy_feature_vectors.csv"
image_path = "/home/alex/Desktop/GTI770-tp2/csv/images/"
# separt la matrice de date en 2 matrice
#on doit avoir au moin 2 matrice vide créer avant l'appel de la méthode

def SplitData(Originaldata,ratio,XTrain=[], Xtest=[]):
    for x in range(Originaldata):
        if random.random()< ratio:
            XTrain.append(x)
        else:
            Xtest.append(x)


#permet de trouver les similarite
def EucliDistance(ins1,ins2,L):
    distance=0
    for x in range(L):
        distance += pow((ins1[x] - ins2[x]),2)
            
    return math.sqrt(distance)
    
    

## k = le nombre d'instance similaire desirer ex si on met 4 on prend les 4 instance les plus proche
#calcule la distance entre k voisin pour avori le plus procher
# a besion d'un matrice de train et teste
def GetVoisin (Xtrain, Xtest,k):
    distance=[]
    l = len(Xtest)-1
    for x in range(len(Xtrain)):
        dis= EucliDistance(Xtrain,Xtest,l)
        distance.append((Xtrain[x],dis))
    distance.sort(key=operator.itemgetter(1))
    ResultVoisin=[]
    for x in range(k):
        ResultVoisin.append(distance[x][0])
    return  ResultVoisin


## obtien la reonse selon un vote entre les donner
def GetReponse(Resultvoisin):
       Vote=[]
       for x in range(len(Resultvoisin)):
        rep = Resultvoisin[x][-1]
        if rep in Vote:
          Vote[rep]+=1
        else:
           Vote[rep] =1
            
        SortVote= sorted(Vote.iteritems(), key=operator.itemgetter(1), reverse=True)
        return SortVote[0][0]



#accuracy pour KNN
## Param
## test:
##def getKnnAccuracy(Test, pred):
    ##int = 0
    ##for x in range(len(Test)):
    ##    if Test[x][-1] is pred[x]:
     ##       int +=1

        
##Appel toute les autre méthode
##param
## DataOriginal la liste de soit les feature des galaxy ou les feature des pouriel
## ratio ratior auquel on sépare les variable de Data original le ratio doit etre 0<ratio<1
## on recommander un ration entre 0.5 et 0.7
## K parametre K pour calculer le K voisin les plus proche ( le paramettre doit etre entre 1<k<51
## retourn une liste avec les résultat et imprime celle-ci
def KNN(DataOriginal,ratio,k):
    X_train=[]
    X_test=[]
    SplitData(DataOriginal,ratio,X_train,X_test)
    Voisin= GetVoisin(X_train,X_test,k)
    Result= GetReponse(Voisin)
    print(Result)


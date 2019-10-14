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
########################################   Initialisations   ########################################

#image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
#image_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/images/'
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"
#dataset_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_label_data_set.csv'
dataset_path = "/home/alex/Desktop/GTI770-tp2/csv/galaxy/galaxy_feature_vectors.csv"
image_path = "/home/alex/Desktop/GTI770-tp2/csv/images/"
# separt la matrice de date en 2 matrice
Def SplitData(Originaldata, ratio, XTrain=[], Xtest=[]):
    for x in range(Originaldata):
        if random.random()< ratio:
            XTrain.append(x)
        else 
            Xtest.append(x)

#permet de trouver les similarite
Def EucliDistance(ins1,ins2,L):
        distance=0
        for x in range(l):
            distance += pow((ins1[x] = ins2[x]),2)
            
        return math.sqrt(distance)    
    
    

## k = le nombre d<istance similaire desirer ex si on met 4 on prend les 4 instance les plus proche
#calcule la distance entre k voisin pour avori le plus procher
# a besion d<un matrice de train et teste
Def GetVoisin (Xtrain, Xtest,k):
    distance=[]
    l = len(Xtest)-1
    for x in range(len(Xtrain)):
        dis= EucliDistance(Xtrain,Xtest,l)
        distance.append((Xtrain[x],dist))
    distance.sort(key=operator.itemgetter(1))
    Resultvoisin=[]
    for x in range(k):
        ResultVoisin.append(Distance[x][0])
    return  Resultvoisin  
## obtien la reonse selon un vote entre les donner    
##Def GetReponse(Resultvoisin):
  ##  Vote=[]
    ##for x in range(len(Resultvoisin)):
      ##  rep = Resultvoisin[x][-1]
        ##if rep in Vote:
          ##  Vote[rep]+=1
        ##else:
          ##  Vote[rep] =1
            
    

        




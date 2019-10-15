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

import csv
from zoo_tree import zoo_tree
from sklearn.model_selection import train_test_split
import numpy as np
########################################   Initialisations   ########################################
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"

# Nombre d'images total du dataset (training + testing)
nb_img = 10
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8


X=[]
Y=[]

########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        #print(type(features),type(galaxy_class))

        X.append(features)
        Y.append(galaxy_class)
        
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,random_state=1, stratify=Y)  # 70% training and 30% test

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# stratify ok
s = np.size(Y_train)
s0_train = np.size(np.where(Y_train ==0))
s0_test = np.size(np.where(Y_train ==0))
print(s0_train/s, s0_test/s)

zoo_tree(X_train, X_test, Y_train, Y_test, ratio_train)


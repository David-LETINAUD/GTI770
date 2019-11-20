#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 4 - Développement d’un système intelligent

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

from functions import *
from SVC_model import PCA_Find_ncomp,PCA_transform,SVM_Gridsearch,SVC_Linear,SVC_rbf
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


SSD_path= "/home/ens/AN03460/Desktop/TP4/music/music/tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"

X, Y = get_data(SSD_path)
X = preprocessing.normalize(X, norm='max',axis = 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=60, stratify=Y)


N_comp=PCA_Find_ncomp(X_train,0.9)
print(N_comp)


PCA_X_Train,PCA_X_Test=PCA_transform(X_train,X_test,N_comp)

Grid= SVM_Gridsearch(PCA_X_Train,Y_train)



print('best param')
print(Grid.best_params_)
print('best score')
print(Grid.best_score_)

SVC_Linear(PCA_X_Train, Y_train, PCA_X_Test, Y_test,1)
SVC_rbf(PCA_X_Train, Y_train, PCA_X_Test, Y_test,10,1)
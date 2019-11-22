#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Course :
    
    GTI770 — Systèmes intelligents et apprentissage machine
    
    Project :
    
    Lab # 4 — Développement d'un système intelligent
    
    Students :
    
    Alexendre Bleau — BLEA14058906
    
    David Létinaud  — LETD05129708
    Thomas Lioret   — LIOT20069605
    
    Group :
    
    GTI770-A19-01
    """

#Imports
from functions import *
from RF_model import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import time
from sklearn import preprocessing


#Ouverture
path_SD = "./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv"   #dataset SpectralDerivate
path_MFC = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv"        #dataset MFC
path_SSD = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"                    #dataset SSD
path_jmirderivatives = "./tagged_feature_sets/msd-jmirderivatives_dev/msd-jmirderivatives_dev.csv"
path_jmirlpc = "./tagged_feature_sets/msd-jmirlpc_dev/msd-jmirlpc_dev.csv"
path_jmirmoments = "./tagged_feature_sets/msd-jmirmoments_dev/msd-jmirmoments_dev.csv"
path_marsyas = "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"
path_mvd = "./tagged_feature_sets/msd-mvd_dev/msd-mvd_dev.csv"
path_rh = "./tagged_feature_sets/msd-rh_dev_new/msd-rh_dev_new.csv"
path_trh = "./tagged_feature_sets/msd-trh_dev/msd-trh_dev.csv"


path_list = [path_marsyas]

X,Y = get_data(path_marsyas) #MAJ GET DATA : NEW sortie

X = X[:100]
Y = Y[:100]
X = preprocessing.normalize(X, norm ='max',axis=0)


#res = RF_dataset_study(path_list,5,5)
#print(res)

#Etude des hyperparamètres

#Nombre d'arbres
list_estimators = [2,5]
n_splits = 5
res = RF_nbEstimators(X,Y,list_estimators,n_splits)
acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"nombre d'estimateurs")
#print(res)


#Profondeur des arbres
list_max_depth = [5,None]
res = RF_maxDepth(X,Y,list_max_depth,n_splits = 5)

acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"profondeur")
#print(res)


#Séparation noeud interne
list_min_samples_splits = [2,3]
res = RF_sampleSplit(X,Y,list_min_samples_splits,n_splits = 5)
acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"internal node")
#print(res)


#Séparation noeud terminal
list_min_samples_leaf = [1,2]
res = RF_sampleLeaf(X,Y,list_min_samples_leaf,n_splits = 5)
acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"leaf node")
#print(res)

#Meilleur K-fold cross validation
list_nb_Ksplit = [5,7]
res = RF_Kfold_Split(X,Y,list_nb_Ksplit)
acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"split")
#print(res)




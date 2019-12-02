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
#from imblearn.under_sampling import RandomUnderSampler
import pickle


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

X,Y,Label = get_data(path_marsyas)
X = preprocessing.normalize(X, norm ='max',axis=0)

#X = X[:100]
#Y = Y[:100]
#rus = RandomUnderSampler(sampling_strategy='auto')
#X,Y = rus.fit_resample(X,Y)



#res = RF_dataset_study(path_list,5,5)
#print(res)

#Etude des hyperparamètres

#Nombre d'arbres
"""
list_estimators = [2,5]
n_splits = 5
res = RF_nbEstimators(X,Y,list_estimators,n_splits)
acc = res[:,0]
f1 = res[:,1]
train_delay = res[:,2]
test_delay = res[:,3]

plot_perf_delay(acc, f1, train_delay,test_delay,"nombre d'estimateurs")
print(res)


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
"""



#############Saving models with pickle


#Marsyas
# Fit the model on training set
X,Y,Label = get_data(path_marsyas)
X = preprocessing.normalize(X, norm ='max',axis=0)
X = X[:100]
Y = Y[:100]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=10,max_depth=10,n_jobs=-1,min_samples_split=2,min_samples_leaf=1)
rfc.fit(X_train, Y_train)

# save the model to disk
pickle.dump(rfc, open('rfc_marsyas.sav', 'wb'))
loaded_model = pickle.load(open('rfc_marsyas.sav', 'rb'))
#print(loaded_model)
result = loaded_model.score(X_test, Y_test)
yo = loaded_model.predict_proba(X_test)
print(yo)
print(len(yo[1]))
print(result)
print(loaded_model)


"""
#path_SSD
X,Y,Label = get_data(path_SSD)
X = preprocessing.normalize(X, norm ='max',axis=0)
X = X[:100]
Y = Y[:100]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=10,max_depth=10,n_jobs=-1,min_samples_split=2,min_samples_leaf=1)
rfc.fit(X_train, Y_train)

# save the model to disk
pickle.dump(rfc, open('rfc_ssd.sav', 'wb'))
loaded_model = pickle.load(open('rfc_ssd.sav', 'rb'))
#print(loaded_model)
result = loaded_model.score(X_test, Y_test)
print(loaded_model.predict(X_test))
print(result)


#path_MFC
X,Y,Label = get_data(path_MFC)
X = preprocessing.normalize(X, norm ='max',axis=0)
X = X[:100]
Y = Y[:100]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=10,max_depth=10,n_jobs=-1,min_samples_split=2,min_samples_leaf=1)
rfc.fit(X_train, Y_train)

# save the model to disk
pickle.dump(rfc, open('rfc_mfc.sav', 'wb'))
loaded_model = pickle.load(open('rfc_mfc.sav', 'rb'))
#print(loaded_model)
result = loaded_model.score(X_test, Y_test)
print(loaded_model.predict(X_test))
print(result)
"""



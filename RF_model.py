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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import time

#Etude des datasets

def RF_dataset_study(path_list,n_estimators = 10,n_splits = 5):

    """
    Fonciton qui permet de faire l'étude des hyperparamètres liés à la Random Forest.
    
    intput:
    list_path (list) : chemin vers le dataset à tester
    n_estimator (int): nombre d'estimateurs (arbres) pour les tests. (default = 10)
    n_splits (int) : nombre de groupe de valeurs pour la cross validation. (default = 5)
    n_jobs (int) : nombre d'opérations effectuées en parallèle.


    output:
    results (ndarray) : liste contenant le nom du dataset avec les accuracy et f1_score associés.
    """
    
    results = []
    rfc = RandomForestClassifier(n_estimators,n_jobs = -1,max_depth = None)

    for path in path_list:

        print(path)

        X,Y,id,le = get_data(path)
        scores = []
        accuracy = []
        kf = KFold(n_splits)
   
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            Y_pred = rfc.predict(X_test)
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))

        m_acc = np.mean(accuracy)
        m_f1 = np.mean(scores)
        results.append((path,[m_acc,m_f1]))

    

    return results




#Etude des hyperparamètres

def RF_nbEstimators(X,Y,list_estimators,n_splits = 5):

    """
    Fonction qui permet de déterminer le nombre d'estimateur maximisant les performances en terme d'accuracy et f1_score.
    
    input:
    X (ndarray) : tableau des primitives
    Y (ndarray) : tableau des étiquettes
    list_estimators (list)   : liste contenant les estimateurs (le nombre d'arbres) à tester dans la forêt. 
    n_splits (int) : nombre de groupe de valeurs pour la cross validation. (default = 5)
    
    output:
    result (ndarray) : Liste contenant l'accuracy, le f1_score, le temps d'entrainement et de prédiciton.
    
    """


    results = []
    kf = KFold(n_splits)

    
    for esti in list_estimators:

        rfc = RandomForestClassifier(esti,n_jobs = -1,max_depth = None)
        scores = []
        accuracy = []
        t1, t2, t3 = 0,0,0
        t_learn, t_pred = 0,0

        for train_index, test_index in kf.split(X):
            t1 = time.time()
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            t2 = time.time()
            Y_pred = rfc.predict(X_test)
            t3 = time.time()
            t_learn += t2 - t1
            t_pred += t3 - t2
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


        results.append([np.mean(accuracy),np.mean(scores),t_learn,t_pred])

    results = np.array(results)

    return results



def RF_maxDepth(X,Y,list_max_depth,n_splits = 5):

    """
    Fonction qui permet de déterminer la profondeur maximisant les performances en terme d'accuracy et f1_score.
    
    input:
    X (ndarray) : tableau des primitives
    Y (ndarray) : tableau des étiquettes
    list_max_depth      (list)   : liste contenant les profondeurs d'arbre à tester.
    n_splits (int) : nombre de groupe de valeurs pour la cross validation. (default = 5)
    
    output:
    result (ndarray) : Liste contenant l'accuracy, le f1_score, le temps d'entrainement et de prédiciton.
    """



    results = []
    kf = KFold(n_splits)
    
    
    for depth in list_max_depth:
        
        rfc = RandomForestClassifier(n_estimators = 10,n_jobs = -1,max_depth = depth)
        scores = []
        accuracy = []
        t1, t2, t3 = 0,0,0
        t_learn, t_pred = 0,0
        
        for train_index, test_index in kf.split(X):
            t1 = time.time()
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            t2 = time.time()
            Y_pred = rfc.predict(X_test)
            t3 = time.time()
            t_learn += t2 - t1
            t_pred += t3 - t2
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
        
        
        results.append([np.mean(accuracy),np.mean(scores),t_learn,t_pred])

    results = np.array(results)
    
    return results



def RF_sampleSplit(X,Y,list_min_samples_splits,n_splits = 5):


    """
    Fonction qui permet de déterminer le nombre minimum d'exemples pour faire la séparation d'un noeud interne maximisant les performances en terme d'accuracy et f1_score.
    
    input:
    X (ndarray) : tableau des primitives
    Y (ndarray) : tableau des étiquettes
    list_min_samples_splits (list)   : liste contenant les nombres minimums d'exemples pour faire la séparation d'un noeud interne.
    n_splits (int) : nombre de groupe de valeurs pour la cross validation. (default = 5)
    
    output:
    result (ndarray) : Liste contenant l'accuracy, le f1_score, le temps d'entrainement et de prédiciton.
    """



    results = []
    kf = KFold(n_splits)
    
    
    for mss in list_min_samples_splits:
        
        rfc = RandomForestClassifier(n_estimators = 10,n_jobs = -1,max_depth = None, min_samples_split = mss)
        scores = []
        accuracy = []
        t1, t2, t3 = 0,0,0
        t_learn, t_pred = 0,0
        
        for train_index, test_index in kf.split(X):
            t1 = time.time()
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            t2 = time.time()
            Y_pred = rfc.predict(X_test)
            t3 = time.time()
            t_learn += t2 - t1
            t_pred += t3 - t2
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
        
        
        results.append([np.mean(accuracy),np.mean(scores),t_learn,t_pred])

    results = np.array(results)

    return results


def RF_sampleLeaf(X,Y,list_min_samples_leaf,n_splits = 5):

    """
    Fonction qui permet de déterminer le nombre minimum d'exemples pour faire la séparation d'un noeud terminal maximisant les performances en terme d'accuracy et f1_score.
    
    
    input:
    X (ndarray) : tableau des primitives
    Y (ndarray) : tableau des étiquettes
    list_min_samples_leaf   (list)   : liste contenant les nombres minimums d'exemples pour faire la séparation d'un noeud terminal.
    n_splits (int) : nombre de groupe de valeurs pour la cross validation. (default = 5)
    
    output:
    result (ndarray) : Liste contenant l'accuracy, le f1_score, le temps d'entrainement et de prédiciton.
    """

    results = []
    kf = KFold(n_splits)
    
    
    for msl in list_min_samples_leaf:
        
        rfc = RandomForestClassifier(n_estimators = 10,n_jobs = -1,max_depth = None, min_samples_leaf = msl)
        scores = []
        accuracy = []
        t1, t2, t3 = 0,0,0
        t_learn, t_pred = 0,0
        
        for train_index, test_index in kf.split(X):
            t1 = time.time()
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            t2 = time.time()
            Y_pred = rfc.predict(X_test)
            t3 = time.time()
            t_learn += t2 - t1
            t_pred += t3 - t2
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
        
        
        results.append([np.mean(accuracy),np.mean(scores),t_learn,t_pred])

    results = np.array(results)

    return results


def RF_Kfold_Split(X,Y,list_nb_Ksplit):

    """
    Fonction permettant de déterminer le meilleur apprentissage en terme de k-split.
    
    input:
    X (ndarray) : tableau des primitives
    Y (ndarray) : tableau des étiquettes
    list_nb_Ksplit (list) : liste contenant le nombre de d'ensemble à tester.
    
    output:
    result (ndarray) : Liste contenant l'accuracy, le f1_score, le temps d'entrainement et de prédiciton.
    """


    results = []
    
    
    
    for ns in list_nb_Ksplit:

        kf = KFold(ns)
        
        rfc = RandomForestClassifier(n_estimators = 10,n_jobs = -1,max_depth = None)
        scores = []
        accuracy = []
        t1, t2, t3 = 0,0,0
        t_learn, t_pred = 0,0
        
        for train_index, test_index in kf.split(X):
            t1 = time.time()
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            rfc.fit(X_train,Y_train)
            t2 = time.time()
            Y_pred = rfc.predict(X_test)
            t3 = time.time()
            t_learn += t2 - t1
            t_pred += t3 - t2
            scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
            accuracy.append(metrics.accuracy_score(Y_test, Y_pred))
        
        
        results.append([np.mean(accuracy),np.mean(scores),t_learn,t_pred])

    results = np.array(results)
    
    return results
    
    
    
    
    

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 3 — Machines à vecteur de support et réseaux neuronaux

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import time
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix,f1_score


"""
    Fonction GridSearchCv qui permet de trouver les meilleurs hyperparamètres    
    input :
        X_train: Liste des vecteurs à analysé
        Y_train: liste de la classification des vecteurs 
    output:
        Grid: Résultat de la fonction gridsearch
"""
def GridSearch_bestparam(X_train,Y_train):
    print('ca commence')

    param = [{'C':[0.001,0.1,1,10],'kernel':['linear']},
             {'C': [0.001,0.1,1,10],'gamma':[0.001, 0.1,1,10], 'kernel': ['rbf']}, ]

    acc=make_scorer(accuracy_score)
    f1= make_scorer(f1_score)
    score = {'F1':f1,'Accuracy':acc}

    svc =svm.SVC(gamma= "scale",cache_size=11264)

    clf = GridSearchCV(svc, param, scoring=score, cv=5, refit='Accuracy', return_train_score=True,n_jobs=10)
    clf.fit(X_train,Y_train)

    print('best param')
    print(clf.best_params_)
    print('best score')
    print(clf.best_score_)

    return clf



"""
    Fonction svc linear qui calcule la matrice de confusion selon l'hyperparamètre choisi     
    input :
        X_train: Liste des vecteurs à analysé pour l'entrainement
        Y_train: liste de la classification des vecteurs pour l'entrainement
        X_test : Liste des vecteurs pour à analyser pour le test 
        Y_test : liste de la classification des vecteurs pour le test
        C      : hyperparamètre C
     output: 
        Matrice de confusion
        Rapport de classification  

"""
def SVCLine(X_train, Y_train, X_test, Y_test,C):

    svc_class = svm.SVC(kernel="linear", C=C)
    start_train = time.time()
    svc_class.fit(X_train, Y_train)
    end_train = time.time()
    start_pred = time.time()
    y_pred = svc_class.predict(X_test)
    end_pred = time.time()

    train_time = (end_train - start_train)
    pred_time = (end_pred - start_pred)
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    print('Temps de training:', train_time, 'Temps de prédiction: ', pred_time)
    return confusion_matrix(Y_test, y_pred), classification_report(Y_test, y_pred), train_time, pred_time


"""
    Fonction svc RBF qui calcule la matrice de confusion selon les hyperparamètres choisis     
    input :
        X_train: Liste des vecteurs à analysé pour l'entrainement
        Y_train: liste de la classification des vecteurs pour l'entrainement
        X_test : Liste des vecteurs pour à analyser pour le test 
        Y_test : liste de la classification des vecteurs pour le test
        C      : hyperparamètre C
        gamma  : hyperparamètre gamma
    output: 
          Matrice de confusion
          Rapport de classification    

"""
def SVC_rbf(X_train, Y_train, X_test, Y_test,C,gamma):
    svc_class = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    start_train= time.time()
    svc_class.fit(X_train, Y_train)
    end_train=time.time()
    start_pred=time.time()
    y_pred = svc_class.predict(X_test)
    end_pred=time.time()

    train_time=(end_train-start_train)
    pred_time=(end_pred-start_pred )
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    print('Temps de training:',train_time, 'Temps de prédiction: ',pred_time)
    return confusion_matrix(Y_test, y_pred),classification_report(Y_test, y_pred),train_time,pred_time

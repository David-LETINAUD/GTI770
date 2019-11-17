# %%
# ! /usr/bin/env python3
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
# from skimage import io

 # Import Decision Tree Classifie
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import sklearn.metrics as metrics
# from sklearn.metric import accuracy_score, f1_score
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import operator
import numpy as np
import random
import math
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pandas as pd
#
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,f1_score
import numpy as np
import matplotlib.pyplot as plt

#def Stratified(n_split,size,radom):
 #   # faire un  split # test a 20 %
  #  Split = StratifiedShuffleSplit(n_split=n_split,test_size=size,random_state=random)
   # return Split

#utriliser plus tard de la faacon suivant
# valeur de retour.split(X,Y)
#fair une for pour chaque element si on veut les utiliser



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

"""
def SVCLine(X_train, Y_train, X_test, Y_test,C):

    svc_class = svm.SVC(kernel="linear", C=C)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    return confusion_matrix(Y_test, y_pred), classification_report(Y_test, y_pred)


"""
    Fonction svc RBF qui calcule la matrice de confusion selon les hyperparamètres choisis     
    input :
        X_train: Liste des vecteurs à analysé pour l'entrainement
        Y_train: liste de la classification des vecteurs pour l'entrainement
        X_test : Liste des vecteurs pour à analyser pour le test 
        Y_test : liste de la classification des vecteurs pour le test
        C      : hyperparamètre C
        gamma  : hyperparamètre gamma

"""
def SVC_rbf(X_train, Y_train, X_test, Y_test,C,gamma):
    svc_class = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    return confusion_matrix(Y_test, y_pred),classification_report(Y_test, y_pred)

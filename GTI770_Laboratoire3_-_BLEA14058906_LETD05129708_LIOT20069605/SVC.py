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
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifie
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

sns.set
import sklearn.metrics as metrics
# from sklearn.metric import accuracy_score, f1_score
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from color import center_color,crop_center
# from fourier_transform import fourier_transform
# from binaryPattern import binaryPatterns
# import match
import operator
import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pandas as pd


########################################   Initialisations   ########################################

####path ordi ETS
dataset_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/galaxy/galaxy_feature_vectors.csv"

image_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/images/"

mail_data_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/spam/spam.csv"

### path ordi Alex
#dataset_path = "/home/alex/Desktop/lab3/GTI770-AlexandreBleau_TP3-branch/GTI770_Laboratoire3_-_BLEA14058906_LETD05129708_LIOT20069605/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/home/alex/Desktop/lab3/GTI770-AlexandreBleau_TP3-branch/GTI770_Laboratoire3_-_BLEA14058906_LETD05129708_LIOT20069605/data/csv/galaxy/TP1_features.csv"
nb_img = 160

# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.7

X = []
Y = []

########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))

    # Lecture ligne par ligne
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        # print(type(features),type(galaxy_class))

        X.append(features)
        Y.append(galaxy_class)

############## FIN LECTURE #########################
########################################   Separation galaxy  ########################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,
                                                    random_state=1)  # 70% training and 30% test

########################################   fin separation   ########################################


#
from sklearn import svm

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix,f1_score
import numpy as np
import matplotlib.pyplot as plt

#pyt    % matplotlib inline
C = [0.001, 0.1, 1, 10]
gamma = [0.001, 0.1, 1.0, 10.0]
# kernel=['linear','rbf']

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
    #param = {'kernel':("linear","rbf"), 'C':[0.001,0.1,1,10]}
    ##param= {'C': [0.001, 0.1, 1],'gamma':[0.001, 0.1, 1 ], 'kernel': ['rbf']}
    acc_scorer=  make_scorer(accuracy_score)
    f1_scorer= make_scorer(f1_score)
    score = {'F1':f1_scorer,'Accuracy':acc_scorer}

    #scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
   # param = [{'kernel': ("rbf"), 'gamma': [0.001, 0.1, 1, 10], 'C': [0.001, 0.1, 1, 10]},{'kernel': ( "linear"),'C': [0.001, 0.1, 1, 10]}]
    svc =svm.SVC(gamma= "scale",cache_size=11264)
  #  clf = GridSearchCV(svc,param,scoring=scoring, cv=5,refit = 'AUC', return_train_score=True)
    clf = GridSearchCV(svc, param, scoring=score, cv=5, refit='Accuracy', return_train_score=True,n_jobs=10)
    clf.fit(X_train,Y_train)

    #print("value")
    #print(clf.cv_results_.keys())

    #return clf
    #print('ca fini')
    #pandas dataframe
    #print('grid test')
    print('best param')
    print(clf.best_params_)
    print('best score')
    print(clf.best_score_)


    return clf
    #return clf.cv_results_

    #return sorted(clf.cv_result_.key())


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
# print('Linear')
# SVCLine(X_train, Y_train, X_test, Y_test,10)
# print('rbf')
# SVC_rbf(X_train, Y_train, X_test, Y_test,1,1)

#Grid = GridSearch_bestparam(X_train,Y_train)

#
#

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
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pandas as pd

########################################   Initialisations   ########################################


dataset_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/galaxy/galaxy_feature_vectors.csv"

image_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/images/"

mail_data_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/spam/spam.csv"

nb_img = 100

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

from sklearn.metrics import classification_report, confusion_matrix
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


#finir les modification pe fair eune 2e, methode pour rdf.

def GridSearch_bestparam(X_train,Y_train):
    print('ca commence')
    param = {'kernel':("linear","rbf"), 'C':[0.001,0.1,1,10]}
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
   # param = [{'kernel': ("rbf"), 'gamma': [0.001, 0.1, 1, 10], 'C': [0.001, 0.1, 1, 10]},{'kernel': ( "linear"),'C': [0.001, 0.1, 1, 10]}]
    svc =svm.SVC(gamma= "scale")
    clf = GridSearchCV(svc,param,scoring=scoring, cv=5,refit = 'AUC', return_train_score=True)
    clf.fit(X_train,Y_train)

    #return clf
    print('ca fini')
    #pandas dataframe
    return clf.cv_results_
    #return sorted(clf.cv_result_.key())
        

def SVCLine(X_train, Y_train, X_test, Y_test,C):
    print("test 1")
    svc_class = svm.SVC(kernel="linear", C=C)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))


def SVC_rbf(X_train, Y_train, X_test, Y_test,C,gamma):
    svc_class = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))


#for c in C:
 #   print("Kernel Type kernel",  "valeur c", c)
  #  SVCLine(X_train, Y_train, X_test, Y_test,c)
#for g in gamma:

 #   for c in C:
  #      print("Kernel Type rbf", "valeur c", c, "valeur gamma", g)
   #     SVC_rbf(X_train, Y_train, X_test, Y_test,c,g)
    # %%


#X_train_Data=tf.data.Dataset.from_tensor_slices((X_train,Y_train))
#print(X_train_Data)
Grid = GridSearch_bestparam(X_train,Y_train)
#print(Grid)

df = pd.DataFrame(data=Grid)
print(df)
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
nb_mail = 100
X_mail = []
Y_mail = []

#
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
#pyt    % matplotlib inline
C_tab = [0.001, 0.1, 1, 10]
gamma_tab = [0.001, 0.1, 1.0, 10.0]
# kernel=['linear','rbf']



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


for c in C_tab:
    print("Kernel Type kernel",  "valeur c", c)
    SVCLine(X_train, Y_train, X_test, Y_test,c)
for g in gamma_tab:

    for c in C_tab:
        print("Kernel Type rbf", "valeur c", c, "valeur gamma", g)
        SVC_rbf(X_train, Y_train, X_test, Y_test,c,g)
    # %%
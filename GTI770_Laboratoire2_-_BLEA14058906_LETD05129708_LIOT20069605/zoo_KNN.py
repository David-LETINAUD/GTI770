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
Dl pycharm sur linux
"""
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

sns.set
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

########################################   Initialisations   ########################################

# image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
# image_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/images/'
# dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"
# dataset_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_label_data_set.csv'
dataset_path = "/home/alex/Desktop/GTI770-tp2/csv/galaxy/galaxy_feature_vectors.csv"
image_path = "/home/alex/Desktop/GTI770-tp2/csv/images/"


def accknn(matrice):
    deno = matrice[0][0] + matrice[1][1]

    nume = matrice[0][0] + matrice[0][1] + matrice[1][0] + matrice[1][1]

    acc = (float(deno) / float(nume))
    return acc


def KNN(Xtrain, Xtest, Ytrain, Ytest, k=15):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(Xtrain, Ytrain)
    y_pred = knn.predict(Xtest)
    confusion_matrix(Ytest, y_pred)
    matrice = confusion_matrix(Ytest, y_pred)

    acc_ = accknn(matrice)
    score_ = metrics.f1_score(Ytest, y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    return ([acc_, score_])

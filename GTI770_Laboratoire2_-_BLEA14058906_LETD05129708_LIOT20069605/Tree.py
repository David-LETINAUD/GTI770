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
"""
#from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import Decision Tree Classifier
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def decision_tree(X_train, X_test, Y_train, Y_test, profondeur=None):
    # Création d'un arbre de décision

    clf = tree.DecisionTreeClassifier(max_depth=profondeur)
    clf = clf.fit(X_train, Y_train)
    # plot_tree(clf, filled=True)
    # plt.show()

    # Prévoir la réponse pour l'ensemble de données de test
    Y_pred = clf.predict(X_test)

    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    #print(acc_,score_)
    return([acc_,score_])
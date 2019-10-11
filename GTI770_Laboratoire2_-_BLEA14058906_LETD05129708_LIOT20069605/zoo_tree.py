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
from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import Decision Tree Classifier
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt


def zoo_tree(X, Y,ratio_train = 0.7,):

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test
    
    # Création d'un arbre de décision

    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(X_train,Y_train) 
    # plot_tree(clf, filled=True)
    # plt.show()

    # Prévoir la réponse pour l'ensemble de données de test
    Y_pred = clf.predict(X_test)

    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    
    return acc_


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

# Imports
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing

# sans prétraitement + Gaussian

def bayes_gaussian_noProcess(X_train, X_test, Y_train, Y_test, var_smooth=1e-09):
    """
    Fonction qui calcule l'accuracy et le f1_score d'un dataset en utilisant la méthode de Bayes gaussien sans traitement des données.
    input:
    X_train  (ndarray)  : tableau des features destinées à l'entrainement.
    X_test   (ndarray)  : tableau des features à tester aux tests.
    Y_train  (ndarray)  : tableau des étiquettes associées aux valeurs d'entrainement.
    Y_test   (ndarray)  : tableau des étiquettes pour les valeurs de test.
    output:
    [acc_,score_] (list) : Résultat de l'accuracy et du f1_score sous forme de liste.
    """

    clf = GaussianNB(priors=None, var_smoothing=var_smooth)  # Par défaut : 1e-09
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    return ([acc_, score_])




def bayes_mutltinomial_scaleData(X_train, X_test, Y_train, Y_test, scale=1):
    """
    Fonction qui calcule l'accuracy et le f1_score d'un dataset en utilisant la méthode de Bayes multinomial avec un scale des données.
    input:
    X_train  (ndarray)  : tableau des features destinées à l'entrainement.
    X_test   (ndarray)  : tableau des features à tester aux tests.
    Y_train  (ndarray)  : tableau des étiquettes associées aux valeurs d'entrainement.
    Y_test   (ndarray)  : tableau des étiquettes pour les valeurs de test.
    scale    (int)      : valeur max pour le scale des data. Par défaut scale vaut 1. Doit être strictement positif.
    output:
    [acc_,score_] (list) : Résultat de l'accuracy et du f1_score sous forme de liste.
    """

    scaler = MinMaxScaler(feature_range=(0, scale), copy=True)  # scale des data entre 0 et 1 par défaut.
    X_train_scale = scaler.fit_transform(X_train)  # On scale les data d'entrainement
    X_test_scale = scaler.fit_transform(X_test)  # On scale les data de test
    clf = MultinomialNB()  # Bayes multinomial
    clf = clf.fit(X_train_scale, Y_train)
    Y_pred = clf.predict(X_test_scale)
    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    return ([acc_, score_])

def bayes_multinomial_kbinDiscetization(X_train, X_test, Y_train, Y_test, nb_bins=5):
    """
    Fonction qui calcule l'accuracy et le f1_score d'un dataset en utilisant la méthode de Bayes multinomial avec une discetisation des données. (KBinDiscretizer)
    input:
    X_train  (ndarray)  : tableau des features destinées à l'entrainement.
    X_test   (ndarray)  : tableau des features à tester aux tests.
    Y_train  (ndarray)  : tableau des étiquettes associées aux valeurs d'entrainement.
    Y_test   (ndarray)  : tableau des étiquettes pour les valeurs de test.
    nb_bins  (int)      : valeur qui détermine le nombre d'intervalles pour la répartition des données (5 par défaut). Doit être strictement positif.

    output:
    [acc_,score_] (list) : Résultat de l'accuracy et du f1_score sous forme de liste.
    """

    pre_proc = preprocessing.KBinsDiscretizer(n_bins=nb_bins, encode='ordinal', strategy='uniform').fit(
        X)  # Jouer avec les hypers paramètres
    X_train_pp = pre_proc.transform(X_train)  # preprocessing des data
    X_test_pp = pre_proc.transform(X_test)
    clf = MultinomialNB()
    clf = clf.fit(X_train_pp, Y_train)
    Y_pred = clf.predict(X_test_pp)
    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    return ([acc_, score_])

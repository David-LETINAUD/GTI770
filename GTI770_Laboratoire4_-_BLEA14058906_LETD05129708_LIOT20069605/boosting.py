#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 4 - Développement d’un système intelligent

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

# inspiré de : https://www.python-course.eu/Boosting.php
from functions import *
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from RN_model import RN_model

layer_sizes = [500] # OK
epochs = 100 # OK avec 100
learning_rate = 0.001
batch_size = 500

dropout = 0.5




def boosting(data_path, weights, RN, RF, SVM ):
    X, Y, id, le = get_data(data_path)
    X = preprocessing.normalize(X, norm='max',axis = 0)

    X = X[:1000]
    Y = Y[:1000]

    nb_features = len(X[0])
    nb_classes = max(Y)
    train_size = len(X) - 1 # -1 a cause de l ID

    X_ID  = np.concatenate((X, id.T), axis=1)

    X_train_ID, X_test_ID, Y_train, Y_test = train_test_split(X_ID, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

    X_train = X_train_ID[:, [range(train_size)]]
    X_test  = X_test_ID[:, [range(train_size)]]
    id = X_test_ID[:, [-1]]

    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_train),
                                                 Y_train)

    # RN MODEL
    hist_obj = RN.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), class_weight=class_weights)
    Y_pred_RN = RN.predict(X_test)

    # RF MODEL
    RF.fit(X_train, Y_train)
    Y_pred_RF = RF.predict_proba(X_test)

    #SVM MODEL
    SVM.fit(X_train, Y_train)
    Y_pred_SVM = SVM.predict_proba(X_test)

    Y_pred_one_hot = weights[0] * Y_pred_RN + weights[1] * Y_pred_RF + weights[2]*Y_pred_SVM

    Y_pred = []
    for i in Y_pred_one_hot:
        Y_pred.append(np.argmax(i))

    # return id/Y_pred/acc/f1
    return Y_pred


def run_boosting(data_path_tab, weights_tab, RN_models_path, RF_models_path, SVM_models_path):
    return 0
    # Load_models
    #
    # result = []
    # for data_path,weights in zip(data_path_tab,weights_tab):
    #     r = boosting(data_path,weights,RN,RF,SVM)
    #     result.append(r)



data_path = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv" #=> MLP 30.7%
X, Y, id, le = get_data(data_path)

# Normalise ou autre traitement

id = np.array([id[:1000]])
X = X[:1000]
Y = Y[:1000]

nb_features = len(X[0]) -1# -1 a cause de l ID
nb_classes = max(Y)
train_size = len(X)

print(np.shape(X),np.shape(id))
X_ID  = np.concatenate((X, id.T),axis=1)
print(np.shape(X_ID))

#print(X_ID[:10])

X_train_ID, X_test_ID, Y_train, Y_test = train_test_split(X_ID, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test



X_train = X_train_ID[:, [range(nb_features)]]
X_test  = X_test_ID[:, [range(nb_features)]]
id_train = X_train_ID[:, [-1]]
id_test = X_test_ID[:, [-1]]

print("#################################################")
print(X[:3], X_ID[:3])
print("*************************************************")
print(X_train[:3], X_train_ID[:3] )
print("#################################################")
print(id_train[:3])
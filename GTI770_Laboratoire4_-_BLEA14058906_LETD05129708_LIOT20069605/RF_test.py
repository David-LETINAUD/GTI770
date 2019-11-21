#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Course :
        
GTI770 — Systèmes intelligents et apprentissage machine

Project :

Lab # 4 — Développement d'un système intelligent

Students :

Alexendre Bleau — BLEA14058906

David Létinaud  — LETD05129708                                                                                                                                                                                     
Thomas Lioret   — LIOT20069605

Group :

GTI770-A19-01
"""

#Imports                                                                                                                                                                            
from functions import *

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics

#Ouverture fichier Spectral Derivate
X,Y = get_data("./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv")
#print(X[7],Y[7])

rfc = RandomForestClassifier(n_estimators = 100, max_depth = None, n_jobs = 3)
#n_estimators : nb arbres dans la forêts
#max_depth : profondeur des arbres - None -> gestion auto


scores = []
accuracy = []
kf = KFold(n_splits = 5) 
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    rfc.fit(X_train,Y_train)
    Y_pred = rfc.predict(X_test)
    scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


print("moyenne des accuracy :",np.mean(accuracy),"les accuracy sont de : ",accuracy)
print("moyenne des f1_score :",np.mean(scores),"les f1_scores sont de : ",scores)


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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics


#fonctions

def RFC(n_estimators = 100, max_depth = None, n_jobs = 2):

    """
    Fonction qui permet de définir l'estimateur du Random Forest Classifier.
    
    input:
    n_estimators (int) : définit le nombre d'arbres dans la forêt. (default = 100)
    max_depth    (int) : définit la profondeur maximal des arbres. (default = None => gestion automatique)
    n_jobs       (int) : définit le nombre d'actions faites en parallèle pour accélérer le traitement. (default = 2)
    
    output:
    estimateur
    """
    rfc = RandomForestClassifier(n_estimators, max_depth, n_jobs)

    return rfc

    

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

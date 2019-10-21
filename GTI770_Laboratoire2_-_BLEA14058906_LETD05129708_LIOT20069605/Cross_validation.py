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

import csv
from zoo_tree import zoo_tree


########################################   Initialisations   ########################################                                                                               

dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"

# Nombre d'images total du dataset (training + testing)                                                                                                                              
nb_img = 100

# Pourcentage de données utilisées pour l'entrainement                                                                                                                               
ratio_train = 0.7


X=[]
Y=[]

########################################   Lecture   ########################################                                                                                       \
                                                                                                                                                                                     
# Lecture du fichier CSV                                                                                                                                                            \
                                                                                                                                                                                     
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                       \
                                                                                                                                                                                     
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        #print(type(features),type(galaxy_class))                                                                                                                                   \
                                                                                                                                                                                     

        X.append(features)
        Y.append(galaxy_class)


############## FIN LECTURE #########################  

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import sklearn.metrics as metrics


from sklearn.naive_bayes import GaussianNB #IMPORTER FONCTION DE LA MEILLEURE METHODE

#CHOISIR LA MEILLEURE METHODE EN FONCTION DES RESULTATS
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X) #X devient un ndarray
Y = np.array(Y)
clf = GaussianNB(priors=None, var_smoothing=1e-09) #REMPLACER PAR MEILLEUR METHODE 
############

scores = []
accuracy = []
kf = KFold(n_splits=10) #K=10 dans l'énoncé

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


print("moyenne des accuracy :",np.mean(accuracy),"les accuracy sont de : ",accuracy)
print("moyenne des f1_score :",np.mean(scores),"les f1_scores sont de : ",scores)

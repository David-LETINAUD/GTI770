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

########################################   Lecture   ########################################                                                                                        
# Lecture du fichier CSV                                                                                                                                                             
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                        
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        #print(type(features),type(galaxy_class))                                                                                                                                    

        X.append(features)
        Y.append(galaxy_class)


############## FIN LECTURE #########################


# Imports                                                                                                                                                 
import numpy as np

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler                                                                                                           

from sklearn import preprocessing



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test

#sans prétraitement + Gaussian
# Création d'un arbre de décision                                                                                                                                                
clf = GaussianNB()
clf = clf.fit(X_train,Y_train)
# Prévoir la réponse pour l'ensemble de données de test                                                                                                                          
Y_pred = clf.predict(X_test)
acc_ = metrics.accuracy_score(Y_test, Y_pred)
print("Accuracy zoo_tree:",acc_)
print("ok")


#scale data + multinomial bayes
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train) #On scale les data d'entrainement
X_test_scale = scaler.fit_transform(X_test) #On scale les data de test

clf = MultinomialNB()
clf = clf.fit(X_train_scale,Y_train)

Y_pred = clf.predict(X_test_scale)
acc_ = metrics.accuracy_score(Y_test,Y_pred)
print("Accuracy zoo_tree:",acc_)
print("ok")
                                                                                                 

#K-Bins discretization + multinomial bayes  
pre_proc = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform').fit(X)
X_train_pp = pre_proc.transform(X_train)
X_test_pp = pre_proc.transform(X_test)

clf = MultinomialNB()
clf = clf.fit(X_train_pp,Y_train)

Y_pred = clf.predict(X_test_pp)
acc_ = metrics.accuracy_score(Y_test,Y_pred)
print("Accuracy zoo_tree:",acc_)
print("ok")




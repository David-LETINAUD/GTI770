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
from Tree import decision_tree


########################################   Initialisations   ########################################                                                                               

dataset_path_galaxie = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"
mail_data_path = "/Users/thomas/Desktop/COURS_ETS/gti770/tp1/git_tp1/GTI770/GTI770_Laboratoire2_-_BLEA14058906_LETD05129708_LIOT20069605/spam.csv"
# Nombre d'images total du dataset (training + testing)                                                                                                                              
nb_img = 100
nb_mail = 100
# Pourcentage de données utilisées pour l'entrainement                                                                                                                               
ratio_train = 0.7


X=[]
Y=[]

########################################   Lecture   ########################################                                                                                       \
                                                                                                                                                                                     
# Lecture du fichier CSV                                                                                                                                                            \
                                                                                                                                                                                     
with open(dataset_path_galaxie, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                       \
                                                                                                                                                                                     
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        #print(type(features),type(galaxy_class))                                                                                                                                   \
                                                                                                                                                                                     

        X.append(features)
        Y.append(galaxy_class)


X_mail=[]
Y_mail=[]  
    
########################################   Lecture Spam   ######################################## 
with open(mail_data_path, 'r') as f:
    mail_features_list = list(csv.reader(f, delimiter=','))
    #print(np.shape(mail_features_list))

    # Lecture ligne par ligne                                                                                                                                                        
    for k in range(nb_mail):
        mail_features = [float(i) for i in mail_features_list[0][0:57]]
        mail_class = int(float( mail_features_list[0][57]))
        mail_features_list.pop(0)
                                                                                                                                   

        X_mail.append(mail_features)
        Y_mail.append(mail_class)
        #print(X_mail)
        #print("--------------Ymail--------------")
        #print( Y_mail)


############## FIN LECTURE SPAM ######################### 

############## FIN LECTURE #########################  

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import sklearn.metrics as metrics


from sklearn.naive_bayes import GaussianNB #IMPORTER FONCTION DE LA MEILLEURE METHODE

#CHOISIR LA MEILLEURE METHODE EN FONCTION DES RESULTATS
#Galaxie
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X) #X devient un ndarray
Y = np.array(Y)
clf = GaussianNB(priors=None, var_smoothing=1e-09) #REMPLACER PAR MEILLEUR METHODE 


#profondeur à changer = 10

    clf = tree.DecisionTreeClassifier(max_depth=profondeur)
    clf = clf.fit(X_train, Y_train)
    # plot_tree(clf, filled=True)                                                                                                                                                    
    # plt.show()                                                                                                                                                                     

    # Prévoir la réponse pour l'ensemble de données de test                                                                                                                          
    Y_pred = clf.predict(X_test)

    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

############
#Mail
X_mail = scaler.fit_transform(X_mail) #X devient un ndarray
Y_mail = np.array(Y)

#profondeur à changer = 5
clf = tree.DecisionTreeClassifier(max_depth=profondeur)

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
print("moyenne des f1_score(galaxie :",np.mean(scores),"les f1_scores sont de : ",scores)




for train_index, test_index in kf.split(X_mail):
    #print("TRAIN:", train_index, "TEST:", test_index)                                                                                                                               
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    scores.append(metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None))
    accuracy.append(metrics.accuracy_score(Y_test, Y_pred))


print("moyenne des accuracy(mail) :",np.mean(accuracy),"les accuracy sont de : ",accuracy)
print("moyenne des f1_score(mail) :",np.mean(scores),"les f1_scores sont de : ",scores)

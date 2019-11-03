#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 3 — Machines à vecteur de support et réseaux neuronaux

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
#import tensorflow as tf
from RN_model import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, average_precision_score
import time
import matplotlib.pyplot as plt
########################################   Initialisations   ########################################
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"

TP1_features_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/TP1_features.csv"
#TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"

# Nombre d'images total du dataset (training + testing)
#nb_img = 16000
nb_img = 1600
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8

X=[]
Y=[]
########################################  Functions
def perf_mesure(y_pred, y_test):
    #f1 = f1_score(y_pred, y_test, average='weighted')
    acc = accuracy_score(y_pred, y_test)
    val_acc = np.sum( np.absolute(y_pred - y_test))
    #rec = recall_score(y_pred, y_test, average='weighted')  
    return [acc,val_acc]

########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    with open(TP1_features_path, 'r') as f_TP1:
        TP1_features_list = list(csv.reader(f_TP1, delimiter=','))
        features_list = list(csv.reader(f, delimiter=','))

        # Recuperation des numéros des images dans l'ordre généré par le TP1
        TP1_features_list_np = np.array(TP1_features_list)[:,0]

        # Lecture ligne par ligne
        for c in range(nb_img):
            features = [float(i) for i in features_list[0][1:75]]

            num_img = str(int(float(features_list[0][0])))

            try :
                # Cherche l'index de l'image num_img dans TP1_features_list
                # pour faire correspondre les features du TP1 avec les nouveaux features
                index = np.where(TP1_features_list_np==num_img)[0]

                features_TP1 = [float(i) for i in TP1_features_list[index[0]][1:4]]

                # concatenation des features
                features = features_TP1 + features

                galaxy_class = int(float(features_list[0][75]))

                X.append(features)
                Y.append(galaxy_class)
            except :
                print("Image {} not find".format(num_img) )

            features_list.pop(0)
            #print(type(features),type(galaxy_class))

X = preprocessing.normalize(X, norm='l2')
X = np.array(X)
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,random_state=1, stratify=Y)  # 70% training and 30% test

print(X.shape)


layer_sizes = [100, 100, 2]
dropout = 0.5
learning_rate = 0.0005
# model = RN_model(layer_sizes, dropout, learning_rate)
# model.fit(X_train, Y_train, batch_size = 100, epochs = 200)
# print('f1 score: {}'.format(f1_score(Y_test, np.where(model.predict(X_test) > 0.5, 1, 0))))


training_delay_RN = []
predicting_delay_RN = []
perf_RN = []
best_index_RN = 0
best_y_test_RN =  []
history_obj = []
#l_rate_range = np.arange(0.0001,0.04,0.0005) #A garder
# l_rate_range = np.logspace(0.0001, 0.004, 1, endpoint=False)
#l_rate_range = [0.000001,0.00005, 0.0005, 0.001, 0.01, 0.02, 0.03, 0.05]
#l_rate_range = [0.0005,0.0008, 0.001]
l_rate_range = [0.000001,0.00005, 0.0005,0.0008, 0.001,0.003, 0.005, 0.01,0.012]
#l_rate_range = np.arange(0.002,0.04,0.002) #A garder
#l_rate_range = np.arange(0.4,1,0.2)
cpt = 0
best_accuracy_RN = 0
for l_rate in l_rate_range:
    model = RN_model(layer_sizes, dropout, l_rate)
    #### Apprentissage
    start = time.time()
    #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
    hist_obj = model.fit(X_train, Y_train, batch_size = 10, epochs = 60, validation_data=(X_test, Y_test))
    
    end = time.time()
    training_delay_RN.append(end - start)

    history_obj.append( list(hist_obj.history.values()))

    #### Prédiction
    start = time.time()
    
    Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)

    end = time.time()
    predicting_delay_RN.append(end - start)

    # Calcul Perfs
    Y_pred = np.argmax(Y_pred, axis = 1)  # Reshape probas vector TO number of the max proba
    perf = perf_mesure(Y_pred, Y_test)
    perf_RN.append(perf)

    if perf[0]> best_accuracy_RN:
        best_accuracy_RN = perf[0]
        best_index_RN = cpt
        best_y_pred_RN =  Y_pred
    cpt+=1
    print("l_rate : ",l_rate, "perf : ", perf)

# Best Perf :
print("Best accuracy : {} for learning_rate = {}".format(perf_RN[best_index_RN][0] , l_rate_range[best_index_RN] ) )
print("Learning delay : {} | predicting delay = {}".format(training_delay_RN[best_index_RN] , predicting_delay_RN[best_index_RN] ) )

ho = np.array(history_obj)
ho = ho.transpose(1,2,0) 
leg = [str(i) for i in l_rate_range]
title_ = ['loss','acc','val_loss','val_acc']
x_lab = "epochs"

def plot_perf(histo,legende,titre,sous_titre):    
    fig, axs = plt.subplots(2,2)
    plt.suptitle(titre, fontsize=16)
    cpt = 0
    for ax in axs:     
        for ax_i in ax:   
            ax_i.title.set_text(sous_titre[cpt])
            #ax_i.set_xlabel("epochs")
            ax_i.legend(legende)
            ax_i.plot( histo[cpt], 'x--')
            cpt+=1

    plt.show()

plot_perf(ho, leg, "RN : étude du learning_rate ",title_)

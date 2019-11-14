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
import matplotlib.pyplot as plt


########################################   Initialisations   ########################################
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"


#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"

TP1_features_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/TP1_features.csv"
#TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"

# Nombre d'images total du dataset (training + testing)
nb_img = 1600
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8


########################################   Lecture   ########################################
def get_data():
    """
    Lit les données, normalise et découpage du dataset      
    output : 
        (np.ndarray) : X_train, X_test, Y_train, Y_test  
    """
    X=[]
    Y=[]

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

                # Récupération de l'image en format entier
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

    # Normalisation des données
    X = preprocessing.normalize(X, norm='max',axis = 0)

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,random_state=60, stratify=Y)  # 70% training and 30% test
    
    return X_train, X_test, Y_train, Y_test


def plot_perf(histo,legende,titre,sous_titre): 
    """
    Affichage des données finales sous forme d'une grille de tableau     
    input : 
         histo (np.ndarray) :       Contient les performances à chaque epochs pour chaque hyperparamètre
         legende (string list) :    Legende à afficher
         titre (string) :           Titre à afficher
         sous_titre (string list) : Sous titre à afficher pour chaque subplot
    """   
    fig, axs = plt.subplots(2,3)
    plt.suptitle(titre, fontsize=16)
    cpt = 0
    for ax in axs:     
        for ax_i in ax:   
            ax_i.title.set_text(sous_titre[cpt])
            
            ax_i.plot( histo[cpt], '-')
            cpt+=1

    # Affichage unique de la legende 
    plt.legend(legende)
    plt.show()

def plot_delay(train_delay,test_delay,titre):   
    """
    Affichage des données des delais d'entrainement et de test     
    input : 
         train_delay (np.ndarray) : Contient les délais d'entrainement pour chaque hyperparamètre
         test_delay (np.ndarray) :  Contient les délais de tests pour chaque hyperparamètre
         titre (string) :           Titre à afficher
    """  
    fig, axs = plt.subplots(1,2)
    plt.suptitle(titre, fontsize=16)

    axs[0].title.set_text("Training delay")
    axs[0].set_xlabel("hyperparameter")
    axs[0].set_ylabel("time (s)")
    axs[0].plot(train_delay,'x--')

    axs[1].title.set_text("Predicting delay")
    axs[1].set_xlabel("hyperparameter")
    axs[1].set_ylabel("time (s)")
    axs[1].plot(test_delay,'x--')

    plt.show()

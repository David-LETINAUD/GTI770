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

import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd


########################################   Lecture   ########################################
def get_data(dataset_path):
    """
    Lit et retourne les données du fichier csv spécifié en entré  
    input :
        dataset_path (string) : nom du fichier à ouvrir
    output : 
        (np.ndarray) : X, Y
    """

    # Lecture du CSV
    features_list = pd.read_csv(dataset_path, header=None, sep = ',')

    # Get_Label
    le = LabelEncoder()
    Y = le.fit_transform(np.array(features_list.iloc[:,-1]) )   
    X = np.array(features_list.iloc[:,2:-1])

    return X, Y


def plot_perf_epochs(histo,legende,titre,sous_titre): 
    """
    Affichage des données finales sous forme d'une grille de tableau     
    input : 
         histo (np.ndarray) :       Contient les performances à chaque epochs pour chaque hyperparamètre
         legende (string list) :    Legende à afficher
         titre (string) :           Titre à afficher
         sous_titre (string list) : Sous titre à afficher pour chaque subplot
    """   
    fig, axs = plt.subplots(2,2)
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

def plot_perf_delay(acc, f1, train_delay,test_delay,titre):   
    """
    Affichage des performances et des delais d'entrainement et de test en fonction des hyperparamètres   
    input : 
        acc (list ou np.ndarray) : Contient l'accuracy du modèle pour chaque hyperparamètre
        f1 (list ou np.ndarray) : Contient le f1-score du modèle pour chaque hyperparamètre
        train_delay (list ou np.ndarray) : Contient les délais d'entrainement pour chaque hyperparamètre
        test_delay (list ou np.ndarray) :  Contient les délais de tests pour chaque hyperparamètre
        titre (string) :           Titre à afficher
    """   

    fig, axs = plt.subplots(2,2)
    plt.suptitle(titre, fontsize=16)

    axs[0][0].title.set_text("Accuracy")
    #axs[0][0].set_xlabel("hyperparameter")
    axs[0][0].set_ylabel("accuracy")
    axs[0][0].plot(acc,'x--')
    
    axs[0][1].title.set_text("f1-score")
    #axs[0][1].set_xlabel("hyperparameter")
    axs[0][1].set_ylabel("f1-score")
    axs[0][1].plot(f1,'x--')

    axs[1][0].title.set_text("Training delay")
    axs[1][0].set_xlabel("hyperparameter")
    axs[1][0].set_ylabel("time (s)")
    axs[1][0].plot(train_delay,'x--')

    axs[1][1].title.set_text("Predicting delay")
    axs[1][1].set_xlabel("hyperparameter")
    axs[1][1].set_ylabel("time (s)")
    axs[1][1].plot(test_delay,'x--')

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
    
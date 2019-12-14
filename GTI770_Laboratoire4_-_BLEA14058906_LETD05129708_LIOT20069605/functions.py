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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


########################################   Lecture   ########################################
def get_data(dataset_path):
    """
    Lit et retourne les données du fichier csv spécifié en entrée  
    input :
        dataset_path (string) : nom du fichier à ouvrir
    output : 
        (np.ndarray) : X, Y, id
        label_encoder : le
    """

    # Lecture du CSV
    features_list = pd.read_csv(dataset_path, header=None, sep = ',')

    # Get_Label
    le = LabelEncoder()
    Y = le.fit_transform(np.array(features_list.iloc[:,-1]) )   
    X = np.array(features_list.iloc[:,2:-1])
    id= np.array(features_list.iloc[:,1])

    # dataset_size = 10000 #len(X)
    # X = X[:dataset_size]
    # Y = Y[:dataset_size]
    # id = id[:dataset_size]

    return X, Y, id, le

def get_data_whithout_labels(dataset_path):
    """
    Lit et retourne les données du fichier csv (sans labels) spécifié en entrée  
    input :
        dataset_path (string) : nom du fichier à ouvrir
    output : 
        (np.ndarray) : X, id
    """

    # Lecture du CSV
    features_list = pd.read_csv(dataset_path, header=None, sep = ',')

    # Get_Label
    X = np.array(features_list.iloc[:,2:-1])
    id= np.array(features_list.iloc[:,1])

    return X, id


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
    


# Fonction inspirée de : https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix (in %)'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #tmp = unique_labels(y_true, y_pred)
    #classes = classes[tmp]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100 # affichage en % pour une meilleure visibilité
        print("Normalized confusion matrix (in %)")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label',
           # xlim = (-0.5,len(classes)-0.5),
           # ylim = (-0.5,len(classes)-0.5)
    )
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    # fmt = '.2f' if normalize else 'd'
    fmt = '.0f' if normalize else 'd' #pour une meilleure visibilité
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.show()
    return ax

# plot_confusion_matrix(y_test,best_y_pred_SVM,class_names, title="KNN Confusion matrix : K = {}".format(K_range[best_index_KNN]))

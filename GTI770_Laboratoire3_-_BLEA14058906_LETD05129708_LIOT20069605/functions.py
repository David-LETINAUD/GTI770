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
import pandas as pd
from textable import textable
#import tensorflow as tf



########################################   Initialisations   ########################################
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#Data de David

#dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"


#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"

#TP1_features_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/TP1_features.csv"
#Data de david
#TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"
#data de PC alex
dataset_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/galaxy/galaxy_feature_vectors.csv"
TP1_features_path ="/home/ens/AN03460/Desktop/tp3/GTI770-AlexandreBleau_TP3-branch/GTI770_Laboratoire3_-_BLEA14058906_LETD05129708_LIOT20069605/TP1_features.csv"
# Nombre d'images total du dataset (training + testing)
#nb_img = 16000
nb_img = 16000
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8


########################################   Lecture   ########################################
def get_data():
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

    #print(X[0])
    X = preprocessing.normalize(X, norm='max',axis = 0)
    #print(X[0])

    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,random_state=60, stratify=Y)  # 70% training and 30% test
    
    return X_train, X_test, Y_train, Y_test


def plot_perf(histo,legende,titre,sous_titre):    
    fig, axs = plt.subplots(2,3)
    plt.suptitle(titre, fontsize=16)
    cpt = 0
    for ax in axs:     
        for ax_i in ax:   
            ax_i.title.set_text(sous_titre[cpt])
            #ax_i.set_xlabel("epochs")
            #ax_i.set_legend(legende)
            
            ax_i.plot( histo[cpt], '-')#, label=legende)
            #ax_i.legend(loc="upper right")
            cpt+=1

    plt.legend(legende)
    plt.show()

def plot_delay(train_delay,test_delay,titre):    
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

    #plt.legend(legende)
    plt.show()

    """
    Séparation des données en deux listes pour l'utilisation de Gridsearchcv      
    Output : 
            X_Grid: Liste des vecteurs normalisés
            Y_Grid: Liste de classification des vecteurs 
         
    """
def get_data_GridSearch():
    X_Grid = []
    Y_Grid = []
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

                    X_Grid.append(features)
                    Y_Grid.append(galaxy_class)
                except :
                    print("Image {} not find".format(num_img) )

                features_list.pop(0)
                #print(type(features),type(galaxy_class))

    #print(X[0])
    X_Grid = preprocessing.normalize(X_Grid, norm='max',axis = 0)
    #print(X[0])

    X_Grid= np.array(X_Grid)
    Y_Grid = np.array(Y_Grid)
    return X_Grid,Y_Grid
    """
    Affichage des données Linear selon le temps de calcule et la précision en fonction du paramètre C     
    input : 
        Grid: Résultat de la fonction gridsearch
         
    """
def plot_Linear_acc(Grid):
    result = Grid.cv_results_
    df = pd.DataFrame(data=result)
    list_accuracy = []
    list_time = []
    list_Param_C = []
    list_gamma = []
    list_kernel = []
    list_test_acc = []
    list_std_train_acc = []

    for i in range(19):
        list_accuracy.append(df.get_value(i, 35, 'mean_train_Accuracy'))
        list_time.append(df.get_value(i, 0, 'mean_fit_time'))
        list_Param_C.append(df.get_value(i, 4, 'param_C'))
        list_gamma.append(df.get_value(i, 6, 'param_gamma'))
        list_kernel.append(df.get_value(i, 5, 'param_kernel'))
        list_test_acc.append(df.get_value(i, 28, 'mean_test_Accuracy'))
        list_std_train_acc.append(df.get_value(i, 36, 'std_train_Accuracy'))

    plt.plot(list_Param_C[0:4],list_test_acc[0:4],label = "Param C linear" )
    plt.scatter(list_Param_C[0:4],list_test_acc[0:4],c='r', label='Precision pour C')
    plt.xlabel('Param C')
    plt.ylabel('Précision')
    plt.xlim(0.001,10)
    plt.ylim(0,1)
    plt.title('Précision en fonction de C')


    plt.legend()

    plt.show()

    plt.plot(list_Param_C[0:4],list_time[0:4],label = "Temps de calcule linear " )
    plt.scatter(list_Param_C[0:4],list_time[0:4],c='r', label='Temps pour  C')
    plt.ylabel('Temps (S)')
    plt.xlabel('Param C')
    plt.ylim(0, 60)
    plt.xlim(0.001, 10)
    plt.title('Temps de traitement en fonction de  C')
    plt.legend()
    plt.show()


    """
    Affichage des données Rbfselon le temps de calcule et la précision en fonction du paramètre C et Gamma      
    input : 
        Grid: Résultat de la fonction gridsearch

    """
def plot_RBF_acc(Grid):

    result = Grid.cv_results_
    df = pd.DataFrame(data=result)
    list_accuracy = []
    list_time = []
    list_Param_C = []
    list_gamma = []
    list_kernel = []
    list_test_acc = []
    list_std_train_acc = []

    for i in range(19):
        list_accuracy.append(df.get_value(i, 35, 'mean_train_Accuracy'))
        list_time.append(df.get_value(i, 0, 'mean_fit_time'))
        list_Param_C.append(df.get_value(i, 4, 'param_C'))
        list_gamma.append(df.get_value(i, 6, 'param_gamma'))
        list_kernel.append(df.get_value(i, 5, 'param_kernel'))
        list_test_acc.append(df.get_value(i, 28, 'mean_test_Accuracy'))
        list_std_train_acc.append(df.get_value(i, 36, 'std_train_Accuracy'))

    # x_line, y_line = np.meshgrid(list_gamma[5:19], list_Param_C[5:19])
    # z_line = np.tile(list_accuracy[5:19],(len(list_accuracy[5:19]),1))
    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    x_line=list_gamma[5:20]
    y_line= list_Param_C[5:20]
    z_line=list_test_acc[5:20]
    ax.set_xlabel('Param Gamma')
    ax.set_ylabel('Param C')
    ax.set_zlabel('Precision')
    #ax.plot3D(x_line,y_line,z_line, 'gray')


    # ax.plot_surface(x_line,y_line,z_line,cmap='ocean')
    ax.scatter3D(x_line,y_line,z_line,cmap='Green',label='Precision de Gamma et C')
    ax.set_title('Precision en fonction de C et gamma ')

    plt.show()

    # x_line, y_line = np.meshgrid(list_gamma[5:19], list_Param_C[5:19])
    # z_line = np.tile( list_time[5:19], (len( list_time[5:19]), 1))
    #
    x_line=list_gamma[5:20]
    y_line=list_Param_C[5:20]
    z_line=list_time[5:20]
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel('param Gamma')
    ax.set_ylabel('Param C')
    ax.set_zlabel('Temps de traitment en seconde')
    #ax.plot3D(x_line, y_line, z_line, 'gray')
    # ax.plot_surface(x_line, y_line, z_line, cmap='ocean')
    ax.set_title('Temps de traitement en fonction de C et gamma ')
    ax.scatter3D(x_line, y_line, z_line, cmap='Green',label='Temps de traitement en fonction de Gamma et C')
    plt.show()


    """
    Affichage des données Linear et RBF dans un tableau avec les résultats permettant un bonne selection des hyperparamètres     
    input : 
        Grid: Résultat de la fonction gridsearch

    """
def plot_analyse_grille(Grid):
    result = Grid.cv_results_
    df = pd.DataFrame(data=result)
    dfData=df[['param_kernel','param_C','param_gamma','mean_test_F1','std_test_F1','mean_test_Accuracy','std_test_Accuracy','rank_test_Accuracy','mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']]
    dfData= dfData.sort_values(['param_kernel','param_C','param_gamma'],ascending=[True,False,False])
    dfData

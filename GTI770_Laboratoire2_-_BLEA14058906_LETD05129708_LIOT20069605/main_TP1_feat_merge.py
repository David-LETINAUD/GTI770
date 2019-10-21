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
from zoo_KNN import KNN
from zoo_BAYES import *



from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
########################################   Initialisations   ########################################
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"
dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"

#TP1_features_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/TP1_features.csv"
TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"

# Nombre d'images total du dataset (training + testing)
nb_img = 1600
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8


X=[]
Y=[]

########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    with open(TP1_features_path, 'r') as f_TP1:
        TP1_features_list = list(csv.reader(f_TP1, delimiter=','))
        features_list = list(csv.reader(f, delimiter=','))

        # Recuperation des numéros des images dans l'ordre générer par le TP1
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


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,random_state=1, stratify=Y)  # 70% training and 30% test

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


# stratify ok
s = np.size(Y_train)
s0_train = np.size(np.where(Y_train ==0))
s0_test = np.size(np.where(Y_train ==0))
print(s0_train/s, s0_test/s)

# hyperparamètres :
Profondeur = 10
K = 44
Var_smoothing = 1e-12

zt = zoo_tree(X_train, X_test, Y_train, Y_test, Profondeur)
zk = KNN(X_train, X_test, Y_train, Y_test, K)
zb = bayes_gaussian_noProcess(X_train,X_test,Y_train,Y_test, Var_smoothing)

print(zt)
print(zk)
print(zb)


def best_hyper_param(func, X_train, X_test, Y_train, Y_test, list_hyper_param):
    acc_list = []
    f1_list = []
    x_plot = []

    elem_acc = 0
    elem_f1 = 0

    max_acc = 0
    max_f1 = 0

    for hyper_param in list_hyper_param:
        acc_, score_ = func(X_train, X_test, Y_train, Y_test, hyper_param)
        x_plot.append(hyper_param)

        acc_list.append(acc_)
        f1_list.append(score_)

        if acc_ > max_acc:
            elem_acc = hyper_param
            max_acc = acc_
        if score_ > max_f1:
            elem_f1 = hyper_param
            max_f1 = score_

    return max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_list, f1_list

def plot_hyper_param(x_plot, acc_plot, f1_plot, hyper_param) :
    fig, ax = plt.subplots()
    ax.plot(x_plot, acc_plot, "or--", label="accuracy")
    ax.plot(x_plot, f1_plot, "xb--", label="f1_score")
    ax.set(xlabel="hyperparametre : {}".format(hyper_param), ylabel='f1_score et accuracy',
            title='f1_score et accuracy en fonction de hyper_param')
    ax.grid()
    plt.legend()
    plt.show()


list_zt = [None, 3, 5, 10, 30, 50]
list_K = np.arange(1, 50, 2)

list_nbins = np.arange(3, 15, 1)
list_var_smoothing = [i for i in np.linspace(1e-11, 1e-8, 10)]  # On fait varier l'hyperparamètre pour le
list_scaler = [i for i in np.linspace(0.2, 3, 10)]

nb_img = 16000
max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_plot, f1_plot = best_hyper_param(zoo_tree,X_train, X_test, Y_train, Y_test, list_zt)
print("zoo_tree :")
print("    Best acc :", max_acc, elem_acc)
print("    Best f1 : ", max_f1, elem_f1)
plot_hyper_param( x_plot, acc_plot, f1_plot, "Profondeur TREE")

max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_plot, f1_plot = best_hyper_param(KNN,X_train, X_test, Y_train, Y_test, list_K)
print("KNN :")
print("    Best acc :", max_acc, elem_acc)
print("    Best f1 : ", max_f1, elem_f1)
plot_hyper_param( x_plot, acc_plot, f1_plot, "K")

max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_plot, f1_plot = best_hyper_param(bayes_gaussian_noProcess,X_train, X_test, Y_train, Y_test, list_var_smoothing)
print("Bayes gauss no process :")
print("    Best acc :", max_acc, elem_acc)
print("    Best f1 : ", max_f1, elem_f1)
plot_hyper_param( x_plot, acc_plot, f1_plot, "var_smooth")

max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_plot, f1_plot = best_hyper_param(bayes_mutltinomial_scaleData,X_train, X_test, Y_train, Y_test, list_scaler)
print("Bayes multinomial scale :")
print("    Best acc :", max_acc, elem_acc)
print("    Best f1 : ", max_f1, elem_f1)
plot_hyper_param( x_plot, acc_plot, f1_plot, "scale")

max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_plot, f1_plot = best_hyper_param(bayes_multinomial_kbinDiscretization,X_train, X_test, Y_train, Y_test, list_nbins)
print("Bayes Discretization :")
print("    Best acc :", max_acc, elem_acc)
print("    Best f1 : ", max_f1, elem_f1)
plot_hyper_param( x_plot, acc_plot, f1_plot, "nbins")


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

import numpy as np

from zoo_tree import zoo_tree
from zoo_KNN import KNN

# from zoo_BAYES import bayes_gaussian_noProcess
########################################   Initialisations   ########################################
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_feature_vectors.csv"
mail_data_path="/home/alex/Desktop/GTI770-tp2/csv/spam/spam.csv"

# Nombre d'images total du dataset (training + testing)

# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.8


nb_mail=3000
X_mail=[]
Y_mail=[]  
    
########################################   Lecture Spam   ######################################## 
with open(mail_data_path, 'r') as f:
    mail_features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                        
    for c in range(nb_mail):
        mail_features = [float(i) for i in mail_features_list[0][0:57]]
        mail_class = int(float( mail_features_list[0][57]))
        mail_features_list.pop(0)
                                                                                                                                   

        X_mail.append(mail_features)
        Y_mail.append(mail_class)
        #print(X_mail)
        #print("--------------Ymail--------------")
        #print( Y_mail)
        
############## FIN LECTURE SPAM #########################            
        
########################################   Separation mail   ######################################## 
X_train, X_test, Y_train, Y_test = train_test_split(X_mail, Y_mail, train_size=ratio_train, random_state=1) # 70% training and 30% test

############## FIN separation mail#########################

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

"""
    Fonction qui permet le calcul des meilleur hyper param pour chaque methode employe
    
    input :
        func (ndarray) : fonction appler pour le calcule de leur hyper parametre
        X_train  (ndarray)  : tableau des features destinées à l'entrainement.
        X_test   (ndarray)  : tableau des features à tester aux tests.
        Y_train  (ndarray)  : tableau des étiquettes associées aux valeurs d'entrainement.
        Y_test   (ndarray)  : tableau des étiquettes pour les valeurs de test.
        list_hyper_param (nparray): tableau des hyper paramettre
    output : 
        max_acc          : valeur numerique de la valeur la plus haute pour la precision
        max_f1           : valeur numerique de la valeur la plus haute pour le F1_score
        elem_acc         : valeur numerique du parametre qui donne la meilleur precision 
        elem_f1          : valeur numerique du parametre qui donne le meilleur F1_score
        x_plot (nparray) : liste des paramettre  
        acc_list(nparray): liste des valeur de la precision
        f1_list( nparray): liste des maleur pour les F1_score
           
        
    
    """
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
    return max_acc, max_f1, elem_acc, elem_f1, x_plot, acc_list, f1_list
"""
    Fonction qui permet dafficher un graphique avec les valeur pour , parametre, precision et F1_score
    
    input :
        x_plot    (ndarray) : liste des paramettre 
        acc_plot  (ndarray) : tableau des features destinées à l'entrainement.
        f1_plot   (ndarray) : tableau des features à tester aux tests.
        hyper_param    : valeur du hyper parametre .
        
    output : 
      
        
        
        
    
    """
def plot_hyper_param(x_plot, acc_plot, f1_plot, hyper_param) :
    fig, ax = plt.subplots()
    ax.plot(x_plot, acc_plot, "or--", label="accuracy")
    ax.plot(x_plot, f1_plot, "xb--", label="f1_score")
    ax.set(xlabel="hyperparametre : {}".format(hyper_param), ylabel='f1_score et accuracy',
            title='f1_score et accuracy en fonction de hyper_param')
    ax.grid()
    plt.legend()
    plt.show()

# Listes des Hyperparamètres à tester
list_zt = [None, 3, 5, 10, 30, 50]
list_K = np.arange(1, 50, 2)

list_nbins = np.arange(3, 15, 1)
list_var_smoothing = [i for i in np.linspace(1e-11, 1e-8, 10)]  # On fait varier l'hyperparamètre pour le
list_scaler = [i for i in np.linspace(0.2, 3, 10)]

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


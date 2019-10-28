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


den_train = np.size(Y_train)
den_test = np.size(Y_test)
num_train = np.size(np.where(Y_train ==0))
num_test = np.size(np.where(Y_test ==0))
print(num_train/den_train, num_test/den_test)


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


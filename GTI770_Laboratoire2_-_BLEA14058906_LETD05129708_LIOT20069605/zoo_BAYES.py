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

import matplotlib.pyplot as plt
from itertools import product


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test

#sans prétraitement + Gaussian

def bayes_gaussian_noProcess(X_train,X_test,Y_train,Y_test):
    """
    """
    clf = GaussianNB(priors=None, var_smoothing=1e-09) #Priors : probabilité des classes (on les considère égales on y touche pas).
    clf = clf.fit(X_train,Y_train)
    # Prévoir la réponse pour l'ensemble de données de test
    Y_pred = clf.predict(X_test)
    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    return([acc_,score_])

print("yoyoyoyoyooyoyoy")
print(bayes_gaussian_noProcess(X_train,X_test,Y_train,Y_test))
print("yoyoyoyoy")



#Variation et plot des hypers paramètres
list_var_smoothing = [i for i in np.linspace(1e-11,1e-8,10)]
x_plot = []
acc_plot = []
f1_plot = []

for ele in list_var_smoothing:
    x_plot.append(ele)
    clf = GaussianNB(priors=None, var_smoothing=ele)
    clf = clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc_ = metrics.accuracy_score(Y_test, Y_pred)
    acc_plot.append(acc_)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    f1_plot.append(score_)


fig, ax = plt.subplots()
ax.plot(x_plot,acc_plot,"or",label = "accuracy")
ax.plot(x_plot,f1_plot,"xb",label = "f1_score")
ax.set(xlabel='feature_range', ylabel='f1_score et accuracy',title='f1_score et accuracy en fonction de var_smoothing')
ax.grid()
plt.legend()




print("=========================================================================================")



def bayes_mutltinomial_scaleData(X_train,X_test,Y_train,Y_test,scale):
    """
    """
    
    scaler = MinMaxScaler(feature_range = (0, scale), copy=True)
    X_train_scale = scaler.fit_transform(X_train) #On scale les data d'entrainement
    X_test_scale = scaler.fit_transform(X_test) #On scale les data de test
    clf = MultinomialNB()
    clf = clf.fit(X_train_scale,Y_train)
    Y_pred = clf.predict(X_test_scale)
    acc_ = metrics.accuracy_score(Y_test,Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    
    return([acc_,score_])


print("yoyoyoyo")
print(bayes_mutltinomial_scaleData(X_train,X_test,Y_train,Y_test,1))
print("coucou")

list_scaler = [(0,i) for i in np.linspace(0.2,1,10)]
x_plot = []
acc_plot = []
f1_plot = []

for ele in list_scaler:
    scaler = MinMaxScaler(feature_range=ele, copy=True)
    X_train_scale = scaler.fit_transform(X_train) #On scale les data d'entrainement
    X_test_scale = scaler.fit_transform(X_test)
    clf = MultinomialNB()
    clf = clf.fit(X_train_scale,Y_train)
    Y_pred = clf.predict(X_test_scale)
    acc_ = metrics.accuracy_score(Y_test,Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    x_plot.append(ele)
    acc_plot.append(acc_)
    f1_plot.append(score_)
    
fig, ax2 = plt.subplots()
ax2.plot(x_plot,acc_plot,"or",label = "accuracy")
ax2.plot(x_plot,f1_plot,"xb",label = "f1_score")
ax2.set(xlabel='feature_range', ylabel='f1_score et accuracy',title='f1_score et accuracy en fonction de feature_range')
ax2.grid()
plt.legend()



print("=========================================================================================")
                                                                                                 

#K-Bins discretization + multinomial bayes  

def bayes_multinomial_kbinDiscetization(X_train,X_test,Y_train,Y_test,nb_bins):
    """
    """

    pre_proc = preprocessing.KBinsDiscretizer(n_bins = nb_bins, encode='ordinal', strategy='uniform').fit(X) #Jouer avec les hypers paramètres
    X_train_pp = pre_proc.transform(X_train) #preprocessing des data
    X_test_pp = pre_proc.transform(X_test)
    clf = MultinomialNB()
    clf = clf.fit(X_train_pp,Y_train)
    Y_pred = clf.predict(X_test_pp)
    acc_ = metrics.accuracy_score(Y_test,Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    return([acc_,score_])


print("coucou")
print(bayes_multinomial_kbinDiscetization(X_train,X_test,Y_train,Y_test,10))
print("coucou")

list_nbins = np.arange(3,10,1)
x_plot = []
acc_plot = []
f1_plot = []

for bin in list_nbins:
    pre_proc = preprocessing.KBinsDiscretizer(n_bins = bin, encode = "ordinal", strategy = "uniform").fit(X)
    X_train_pp = pre_proc.transform(X_train) #preprocessing des data
    X_test_pp = pre_proc.transform(X_test)
    clf = MultinomialNB()
    clf = clf.fit(X_train_pp,Y_train)
    Y_pred = clf.predict(X_test_pp)
    acc_ = metrics.accuracy_score(Y_test,Y_pred)
    score_ = metrics.f1_score(Y_test, Y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)
    x_plot.append(bin)
    acc_plot.append(acc_)
    f1_plot.append(score_)


#print(acc_plot)
#print(f1_plot)

fig, ax3 = plt.subplots()
ax3.plot(x_plot,acc_plot,"or",label = "accuracy")
ax3.plot(x_plot,f1_plot,"xb",label = "f1_score")
ax3.set(xlabel='nombre de bins', ylabel='f1_score et accuracy',title='f1_score et accuracy en fonction du nombre de bins')
ax3.grid()
plt.legend()
plt.show()

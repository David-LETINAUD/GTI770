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
from color import center_color
from fourier_transform import fourier_transform
from binaryPattern import binaryPatterns
import matplotlib.pyplot as plt

def FeaturesProcess(img,cs_color,th_fft,nr_binaryPattern):
    """
    Fonction qui permet le calcul de chaque features d'img
    
    input :
        img (ndarray) : image quelconque
        cs_color (int) : taille du centre de l'image à prendre en compte pour calculer la moyenne du niveau de gris
        th_fft (int) : seuil à partir duquel on prend en compte les fréquences (strictement positif)
        nr_binaryPattern ([int,int]) : 
                    nr_binaryPattern[0] : nombre de points à prendre en compte sur le périmètre du cercle
                    nr_binaryPattern[1] : taille du rayon du cercle
    output : 
        (list) retourne la liste des features calculées
    
    """
    Features = []

    # Calculs des Features
    f_c = center_color(img,cs_color)
    f_fft = fourier_transform(img,th_fft)
    f_bp = binaryPatterns(img,nr_binaryPattern[0],nr_binaryPattern[1])  

    Features.append(f_c)   
    Features.append(f_fft)
    Features.append(f_bp)

    # Retourne les features calculés
    return Features  
"""
    Fonction qui permet le calcul des meilleur hyper param pour chaque methode employe
    
    input :
        func (ndarray) : fonction appler pour le calcule de leur hyper parametre
        X_train  (ndarray)  : tableau des features destiné à l'entrainement.
        X_test   (ndarray)  : tableau des features destiné aux tests.
        Y_train  (ndarray)  : tableau des étiquettes associé aux valeurs d'entrainement.
        Y_test   (ndarray)  : tableau des étiquettes pour les valeurs de test.
        list_hyper_param (nparray): tableau des hyper paramettre
    output : 
        max_acc          : valeur numerique de la valeur la plus haute pour la précision
        max_f1           : valeur numerique de la valeur la plus haute pour le F1_score
        elem_acc         : valeur numerique du parametre qui donne la meilleur precision 
        elem_f1          : valeur numerique du parametre qui donne le meilleur F1_score
        x_plot (nparray) : liste des paramètre  
        acc_list(nparray): liste des valeurs de la précision
        f1_list( nparray): liste des valeurs pour les F1_score
           
        
    
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

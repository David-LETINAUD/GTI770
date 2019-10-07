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


from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifie
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt

from color import center_color,crop_center
from fourier_transform import fourier_transform
from binaryPattern import binaryPatterns

########################################   Initialisations   ########################################

image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"

# Nombre d'images total du dataset (training + testing)
# Tester les paramètres sur 10% du dataset
nb_img = 1600
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.7
# Taille de rognage de l'image
crop_size = 180

X = [] # Contient les features de l'image
Y = [] # Contient les classes associées aux images 

# Paramètres de chaque features
fft_threshold = 140
color_center_size = 18
bp_calibration = [100,50]

Paramresult = []

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
    
    # plt.imshow(img)
    # plt.show()

    # Calculs des Features
    f_c = center_color(img,cs_color)
    # f_fft = fourier_transform(img,th_fft)
    # f_bp = binaryPatterns(img,nr_binaryPattern[0],nr_binaryPattern[1])  

    Features.append(f_c)   
    # Features.append(f_fft)
    # Features.append(f_bp)

    # Retourne les features calculés
    return Features    

best_accuracy = -1
best_param = -1

########################################   Lecture   ########################################
for color_center_size in range(1,50):
    # Lecture du fichier CSV
    with open(dataset_path) as f:
        f_csv = csv.reader(f)
        next(f_csv) # On passe la 1ere ligne d'entête
        
        # Lecture ligne par ligne
        for ligne,i in zip(f_csv,range(nb_img)):
            
            # Lecture et rognage de l'image
            image = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size)
            # Calcul des features et stockage dans X
            X.append( FeaturesProcess(image, color_center_size, fft_threshold, bp_calibration) )
            # Sauvegarde de la classe correspondante dans Y
            Y.append(1 * (ligne[1]=="smooth"))  # smooth :1 et spiral : 0  

    # Diviser l'ensemble de données en un ensemble d'apprentissage et un ensemble de test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test
    # Création d'un arbre de décision 
    clf = tree.DecisionTreeClassifier()
    # Construit les décision de l'arbre de classification
    clf = clf.fit(X_train,Y_train)
    # Prévoir la réponse pour l'ensemble de données de test
    Y_pred = clf.predict(X_test)
    # Précision du modèle, à quelle fréquence le classificateur est-il correct ?
    accuracy = metrics.accuracy_score(Y_test, Y_pred)

    if accuracy> best_accuracy:
        best_param = color_center_size
        best_accuracy = accuracy
        print(color_center_size,best_accuracy)

    Paramresult.append(accuracy)
    # Pour que l'index du tableau correspondent au seuil testé
    if color_center_size==1:
        Paramresult.append(accuracy)

print(Paramresult)
print(best_param, best_accuracy)
plt.plot(Paramresult,'x')
plt.xlabel('accuracy')
plt.ylabel('fft_threshold')
plt.title('accuracy en fonction de fft_threshold')
plt.grid(True)
plt.show()

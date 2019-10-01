#! /usr/bin/env python3 
# -*- coding: utf-8 -*-
#######   Initialisation
from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifie
import sklearn.metrics as metrics
import cv2

import numpy as np
import csv
import matplotlib.pyplot as plt
import random

from color import center_color
from fourier_transform import fourier_transform
from binaryPattern import *

image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"


# Nombre d'images
nb_img = 200
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.7
# Taille de rognage de l'image
crop_size = 180

X = [] # Contient les features de l'image
Y = [] # Contient les classes associées aux images 

# Paramètre des features
fft_threshold = 150
color_threshold = 11
#bp_calibration =

def FeaturesProcess(img,th_color,th_fft):
    Features = []
    
    # plt.imshow(img)
    # plt.show()

    # Calculs des Features
    f_c = center_color(img,th_color)
    f_fft = fourier_transform(img,th_fft)
    f_bp = binaryPatterns(img)  

    Features.append(f_c)   
    Features.append(f_fft)
    Features.append(f_bp)

    # Retourne les features calculés
    return Features    



########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
    
    t = 1
    # Lecture ligne par ligne
    for ligne,i in zip(f_csv,range(nb_img)):
        
        image = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size)
        X.append(FeaturesProcess(image,color_threshold,fft_threshold))
        Y.append(1 * (ligne[1]=="smooth"))  # smooth :1 et spiral : 0
       

########################################    Entrainement   ########################################
# Diviser l'ensemble de données en un ensemble d'apprentissage et un ensemble de test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test
# print(X_train)
# print(Y_train)
# print(X_test)
# print(Y_test)


# Création d'un arbre de décision 
clf = DecisionTreeClassifier()

# Construit les décision de l'arbre de classification
clf = clf.fit(X_train,Y_train)

# Prévoir la réponse pour l'ensemble de données de test
Y_pred = clf.predict(X_test)

# Précision du modèle, à quelle fréquence le classificateur est-il correct ?
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

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

from color import *  
from fourier_transform import *
from binaryPattern import *

# image_path = "/home/ens/AN03460/Desktop/tp0/data/data/images/"
# dataset_path = "/home/ens/AN03460/Desktop/tp0/data/data/csv/galaxy/galaxy_label_data_set.csv"
image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"

# Nombre d'images de chaque classe
nb_img = 50

ratio_train = 0.7

crop_size = 180

#######   2
Y = [] # Classe
#Features

X_mean_color = []
X_f= []
Paramresult = []
def f_X(img,i):
    global X_f
    global Patern
    Features = []
    th = 150
    global X_mean_color
    m=center_color(img,i)
    # plt.imshow(img)
    # plt.show()

    fft = fourier_transform(img,th)
    e = binaryPatterns(img)  

    Features.append(m)   
    Features.append(fft)

    Features.append(e)
    
    X_f.append(Features)
    #X_mean_color.append(m)

   

def f_smooth(img):
    print("f_smooth")

def f_spirale(img):
    print("f_spirale")


########################################   TRAINING   ########################################
# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
    for i in range(1,50):
        sm=1
        sp=1
    # Lecture ligne par ligne
        for ligne in f_csv:
            if sm<=nb_img and ligne[1]=="smooth":
                X = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size) 
                f_X(X,i)
                Y.append(0)  
                sm=sm+1
            elif sp<=nb_img and ligne[1]=="spiral":
                X = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size) 
                f_X(X,i)
                Y.append(1)  
                sp=sp+1
            elif sm>nb_img and sp>nb_img:
                # Quand toutes les images sont enregistrées => sortir de la boucle
                break
            elif sm<=nb_img and sp<=nb_img:
                print(ligne[1] + " inconnu")

        X =  X_mean_color     
        X_train, X_test, Y_train, Y_test = train_test_split(X_f, Y, train_size=0.7, random_state=1) # 70%     
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,Y_train)
        Y_pred = clf.predict(X_test)
        ParamResult.append(metrics.accuracy_score(Y_test, Y_pred))

########################################   PROCESSING   ########################################
# ESSAYER EN ENLEVANT LE VERT (le rouge et le bleue peuvent être plus discriminant)
color_threshold = np.median(X_mean_color)
#color_threshold = otsu_threshold(X_mean_color)
print(color_threshold)
#print(Features)



########################################    TESTING   ########################################

#features = [color, fft, SIFT]
###X =  X_mean_color #[X_mean_color, nb_freq, ...] # features de chaque image d'entrainement
#Y  # Label ciblé

# Split dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X_f, Y, train_size=0.7, random_state=1) # 70% training and 30% test
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
Y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?

###ParamResult.append(metrics.accuracy_score(Y_test, Y_pred))
###print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print(Paramresult)

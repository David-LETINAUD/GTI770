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

image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"


# Nombre d'images de chaque classe
nb_img = 100

ratio_train = 0.7

crop_size = 180

#######   2
Y = [] # Classe
X = []
#Features

X_mean_color = []
X_f= []
TestParam = []

fft_threshold = 150
color_threshold = 18

def f_X(img,th_color,th_fft):
    Features = []

    m=center_color(img,th_color)
    # plt.imshow(img)
    # plt.show()

    fft = fourier_transform(img,th_fft)
    e = binaryPatterns(img)  

    Features.append(m)   
    Features.append(fft)

    Features.append(e)
    #X_f.append(Features)
    #X_mean_color.append(m)

    return Features
    

########################################   TRAINING   ########################################
# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
    
    t = 1
    # Lecture ligne par ligne
    for ligne,i in zip(f_csv,range(nb_img)):
        
        image = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size)
        X.append(f_X(image,color_threshold,fft_threshold))
        Y.append(1 * (ligne[1]=="smooth"))  # smooth :1 et spiral : 0
       

########################################   PROCESSING   ########################################
# ESSAYER EN ENLEVANT LE VERT (le rouge et le bleue peuvent être plus discriminant)
color_threshold = np.median(X_mean_color)
#color_threshold = otsu_threshold(X_mean_color)
print(color_threshold)
#print(Features)



########################################    TESTING   ########################################
# Split dataset into training set and test set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1) # 70% training and 30% test
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
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))



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
nb_img = 1600

ratio_train = 0.7

crop_size = 180

#######   2
Y = [] # Classe
#Features

X = []
X_f= []


fft_threshold = 6
color_threshold = 11
def f_X(img,th_color,th_fft):
    Features = []

    #m=center_color(img,th_color)
    # plt.imshow(img)
    # plt.show()

    fft = fourier_transform(img,th_fft)
    #e = binaryPatterns(img)  

    #Features.append(m)   
    Features.append(fft)

    #Features.append(e)
    #X_f.append(Features)
    #X_mean_color.append(m)

    return Features

Paramresult = []
accuracy_prec = -1
best_param = -1
best_accuracy = -1
########################################   TRAINING   ########################################
# Lecture du fichier CSV
for fft_threshold in range(1,1000):
    
    with open(dataset_path) as f:
        f_csv = csv.reader(f)
        en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
        
        
        # Lecture ligne par ligne
        for ligne,i in zip(f_csv,range(nb_img)):        
            image = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size)
            X.append(f_X(image,color_threshold,fft_threshold))
            Y.append(1 * (ligne[1]=="smooth"))  # smooth :1 et spiral : 0
                

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=1) # 70%     
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    if accuracy> best_accuracy:
        best_param = fft_threshold
        best_accuracy = accuracy

    Paramresult.append(accuracy)
    if fft_threshold==1:
        Paramresult.append(accuracy)


print(best_param)
plt.plot(Paramresult,'x')
plt.xlabel('accuracy')
plt.ylabel('fft_threshold')
plt.title('accuracy en fonction de fft_threshold')
plt.grid(True)
plt.show()
#print(Paramresult, Paramresult.index(max(Paramresult))+1)

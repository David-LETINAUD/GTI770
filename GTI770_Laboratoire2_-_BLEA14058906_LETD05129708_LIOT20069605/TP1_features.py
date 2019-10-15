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

#! /usr/bin/env python3 
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine
Project :
Lab # 1 — Définition et extraction de primitives
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
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import Decision Tree Classifie
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt

from color import center_color,crop_center
from fourier_transform import fourier_transform
from binaryPattern import binaryPatterns

########################################   Initialisations   ########################################

#image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
#image_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/images/'
dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_label_data_set.csv"
TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"
#dataset_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_label_data_set.csv'
image_path = "/home/ens/AQ38840/Desktop/data/data/images/"


# Nombre d'images total du dataset (training + testing)
nb_img = 16000
# Taille de rognage de l'image
crop_size = 180

TP1_feat_lignes = []
X = [] # Contient les features de l'image
# Paramètres de chaque features determinees au TP1
fft_threshold = 140
color_center_size = 18
bp_calibration = [100,50]

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
    f_fft = fourier_transform(img,th_fft)
    f_bp = binaryPatterns(img,nr_binaryPattern[0],nr_binaryPattern[1])  

    Features.append(f_c)   
    Features.append(f_fft)
    Features.append(f_bp)

    # Retourne les features calculés
    return Features    


########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    next(f_csv) # On passe la 1ere ligne d'entête
    
    # Lecture ligne par ligne
    for ligne,i in zip(f_csv,range(nb_img)):
        l_CSV = []
        # Lecture et rognage de l'image
        image = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size)
        X = FeaturesProcess(image, color_center_size, fft_threshold, bp_calibration)
	
        l_CSV.append(ligne[0])
        l_CSV.append(str(X[0]))
        l_CSV.append(str(X[1]))
        l_CSV.append(str(X[2]))
        l_CSV.append(str(1 * (ligne[1]=="spiral")))
        TP1_feat_lignes.append(l_CSV)

f.close()
########################################   Ecriture   ########################################
print(TP1_feat_lignes)
with open(TP1_features_path, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(TP1_feat_lignes)
    writeFile.close()
       



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

import csv
from color import crop_center
from main_functions import  FeaturesProcess

########################################   Initialisations   ########################################

dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_label_data_set.csv"
image_path = "/home/ens/AQ38840/Desktop/data/data/images/"
# Fichier de sortie
TP1_features_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/TP1_features.csv"


# Taille de rognage de l'image
crop_size = 180

TP1_feat_lignes = []

# Paramètres de chaque features determinees au TP1
fft_threshold = 140
color_center_size = 18
bp_calibration = [100,50]
  


########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    next(f_csv) # On passe la 1ere ligne d'entête
    
    # Lecture ligne par ligne
    for ligne in f_csv:#,i in zip(f_csv,range(nb_img)):
        l_CSV = []
        # Lecture et rognage de l'image
	ID = str(int(float(ligne[0])) # Correction format pour MAC et Windows
        image = crop_center(io.imread( image_path + ID + ".jpg" ),crop_size,crop_size)
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
       



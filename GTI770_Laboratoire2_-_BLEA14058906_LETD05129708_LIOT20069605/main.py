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

########################################   Initialisations   ########################################
dataset_path = "/home/ens/AQ38840/Desktop/data/data/csv/galaxy/galaxy_feature_vectors.csv"

# Nombre d'images total du dataset (training + testing)
nb_img = 10
# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.7


#X_tree
#X_bayes
########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))

    # Lecture ligne par ligne
    for c in range(nb_img):
        features = [float(i) for i in features_list[0]]
        print(type(features))
        print(features)
        print("")
        print("")
        #X_tree.append(zoo_tree(features))
        features_list.pop(0)

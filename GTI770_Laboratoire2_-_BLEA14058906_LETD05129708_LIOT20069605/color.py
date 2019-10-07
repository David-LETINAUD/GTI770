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

import numpy as np

# Rogne de manière centrée l'image img aux dimensions cropx/cropy
# Retourne l'image rognée
def crop_center(img,cropx,cropy):
    """
    Fonction qui permet de rogner, de manière centrée, une image en taille cropx,cropy
    
    input :
        img (ndarray) : image quelconque 
        cropx (int) : nombre de pixels à garder sur l'axe des x
        cropy (int) : nombre de pixels à garder sur l'axe des y
    output : 
        (ndarray) image rognée de taille cropx,cropy
    
    """
    x = img.shape[0] # Sauvegarde la taille de l'image en x
    y = img.shape[1]
    startx = x//2-(cropx//2) # '//' Renvoie la partie décimale du quotient.
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Retourne la couleur moyenne du centre de l'image
# La taille du centre de l'image est déterminée par taille_centre
def center_color(img,taille_centre):
    """
    Fonction qui permet de calculer la couleur moyenne du centre d'une image
    
    input :
        img (ndarray) : taille de l'image quelconque
        taille_centre (int) : taille en pixels du centre de l'image à partir duquel on calcul la moyenne
    output : 
        (int) couleur moyenne du centre de l'image
    
    """

    img_crop = crop_center(img,taille_centre,taille_centre)
    # Conversion en niveaux de gris
    img_grey = 0.2989*img_crop[:,:,0] + 0.5870*img_crop[:,:,1] + 0.1140*img_crop[:,:,2]

    return int(np.mean(img_grey))


     

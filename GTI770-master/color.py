#! /usr/bin/env python3 
# -*- coding: utf-8 -*-
import numpy as np

# Rogne de manière centrée l'image img aux dimensions cropx/cropy
# Retourne l'image rognée
def crop_center(img,cropx,cropy):
    x = img.shape[0] # Sauvegarde la taille de l'image en x
    y = img.shape[1]
    startx = x//2-(cropx//2) # '//' Renvoie la partie décimale du quotient.
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# Plusieurs façons de supprimer les 3 canaux de couleurs pour n'en faire qu'1 gris
# On peut par exemple donner un poids différents à chaque couleur
# img : image à griser 
# Retourne une image grise 
def to_grey(img):
    return 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]

# Retourne la couleur moyenne du centre de l'image
# La taille du centre de l'image est déterminée par taille_centre
def center_color(img,taille_centre):
    """
    Fonction qui permet de calculer la couleur moyenne du centre d'une image
    
    input :
        image = ndarray (taille de l'image quelconque)
        taille_centre = taille en pixel du centre de l'image à partir duquel on calcul la moyenne
    output : 
        couleur moyenne du centre de l'image
    
    """

    img_crop = crop_center(img,taille_centre,taille_centre)
    return int(np.mean(to_grey(img_crop)))


     

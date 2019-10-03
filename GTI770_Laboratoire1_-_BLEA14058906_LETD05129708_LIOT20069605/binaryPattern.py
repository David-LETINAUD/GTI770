
#! /usr/bin/env python3 
# -*- coding: utf-8 -*-
# Inspirer de www.pyimagesearch.com/2015/12/07/local-binary-paterns-with-python-opencv/
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

from skimage import feature
from scipy.stats import entropy as scipy_entropy 
import numpy as np
from numpy import unique
from color import crop_center
import cv2

class GalaxyBinaryPatterns:
  """
   Class binary Paterne. En premier lieu, elle permet de caculer les forme de la surface d'une image, 
   en second lieu elle calcule l'entropy de celle ci 
    
    input :
        numPoint = integer, fournie lkes nombre de point d'interet du relief de l'image
        raduis = rayon en bit de l'image, represente la zone dans laquel on cehrcher les point
        img = image a identifier
        
    output : 
        entropy de chaque image 
    
    """
    def __init__ (self,numPoints, radius):
        # Enregistre les points et radius 
        # Permet la construction d'un histograme
        self.numPoints = numPoints
        self.radius = radius
           
    def Galaxy_description(self,image , eps=1e-7): 
        lbp = feature.local_binary_pattern(image , self.numPoints , self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),bins = np.arange(0 , self.numPoints +3),range=(0,self.numPoints +2))
        hist = hist.astype("float")
        hist/= (hist.sum()+eps)
        
        return hist

def binaryPatterns(img,numPoints,radius):
    """
    Fonction qui permet le calcul du binaryPatterns d'une image selon les paramètres numPoints,radius
    
    input :
        img (ndarray) : image quelconque
        numPoints (int): nombre de points à prendre en compte sur le périmètre du cercle
        radius (int): taille du rayon du cercle
    output : 
        (int) Retourne l'histogramme du binaryPattern (motifs binaires) de l'image 
    
    """
    Patern = GalaxyBinaryPatterns(numPoints,radius)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    Hist = Patern.Galaxy_description(gris)

    _,counts = unique(Hist,return_counts=True)

    return int(100 * scipy_entropy(counts,base=2))
        
    
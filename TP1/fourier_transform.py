#! /usr/bin/env python3 
# -*- coding: utf-8 -*-
import numpy as np

def fourier_transform(image,seuil):
    
    """
    Fonction qui permet de calculer le nombre de fréquences présentes dans une image grâce à 
    la transormée de Fourier d'une image
    
    input :
        image (ndarray) : image quelconque
        seuil (int) : seuil à partir duquel on prend en compte les fréquences (strictement positif)
    output : 
        nombre de fréquences (ndarray de taille (1,Nombre de fréquences supérieures au seuil))
    
    """
   
    f = np.fft.fft2(image) #transformée de Fourier de l'image
    fshift = np.fft.fftshift(f) #on déplace l'échelle des fréquences pour ne pas avoir de fréquences négatives
    magnitude_spectrum = np.abs(fshift) #on calcule la valeur absolue de l'amplitude des raies spectrales.
    temp_array = np.where(magnitude_spectrum > seuil) #on compare la veleur de l'amplitude par rapport au seuil  
                                                        #et on stocke la valeur dans le tableau si le test est vrai
    #print (np.shape(temp_array))
    res = np.shape(temp_array)[1] #on calcule le nombre de valeur d'amplitude ayant passé le test
    
    return(res)




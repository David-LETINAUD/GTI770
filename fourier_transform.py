#! /usr/bin/env python3 
import numpy as np

def fourier_transform(image,seuil):
    
    """
    Fonction qui permet de calculer le nombre de fréquences présentes dans une image grâce à 
    la transormée de Fourier d'une image
    
    input :
        image = ndarray (taille de l'image quelconque)
        seuil = seuil à partir duquel on prend en compte les fréquences (strictement positif)
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


# cpt = 0
# th = -1 #A définir : doit être > 0 

# from main import *

# list_smooth = []
# list_spiral = []

# for img in X_train_crop:
    
#     tmp = fourier_transform(img,th)
    
#     if cpt in smooth_index:
#         list_smooth.append(tmp)
    
#     if cpt in spiral_index:
#         list_spiral.append(tmp)
    
#     cpt += 1

# list_smooth = np.array(list_smooth)
# list_spiral = np.array(list_spiral)
# print(list_smooth,list_spiral)

# print(np.shape(list_smooth))
# print('la moyenne (smooth) est ', np.mean(list_smooth))
# print('la médiane (smooth) est ', np.median(list_smooth))
# #print(list_smooth)
# print('======================================================================================')
# print(np.shape(list_spiral))
# print('la moyenne (spiral) est ', np.mean(list_spiral))
# print('la médiane (spiral) est ', np.median(list_spiral))
# #print(list_spiral)


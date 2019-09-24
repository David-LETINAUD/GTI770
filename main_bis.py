#######   Initialisation
from skimage import io
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

from color import *  

image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"


# Nombre d'images de chaque classe
nb_img = 50

ratio_train = 0.7

crop_size = 240

#######   2
X_mean_color = []

def f_X(img):
    global X_mean_color
    m=center_color(img)
    X_mean_color.append(m)

def f_smooth(img):
    print("f_smooth")

def f_spirale(img):
    print("f_spirale")



# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
    
    sm=1
    sp=1
    
    # Lecture ligne par ligne
    for ligne in f_csv:
        X = crop_center(io.imread( image_path + ligne[0] + ".jpg" ),crop_size,crop_size) 
        f_X(X)

        if sm<=nb_img and ligne[1]=="smooth":
             
           sm=sm+1
        elif sp<=nb_img and ligne[1]=="spiral":

           sp=sp+1
        elif sm>nb_img and sp>nb_img:
            # Quand toutes les images sont enregistrées => sortir de la boucle
            break
        elif sm<=nb_img and sp<=nb_img:
            print(ligne[1] + " inconnu")


print(X_mean_color)
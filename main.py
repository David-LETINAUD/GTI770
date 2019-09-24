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
X = []
Y = []

# Lecture du fichier CSV
with open(dataset_path) as f:
    f_csv = csv.reader(f)
    en_tetes = next(f_csv) # On passe la 1ere ligne d'entête
    
    sm=1
    sp=1
    
    # Lecture ligne par ligne
    for ligne in f_csv:
        #print(ligne)
        if sm<=nb_img and ligne[1]=="smooth":
           X.append(io.imread( image_path + ligne[0] + ".jpg" ))   
           Y.append('smooth')
           sm=sm+1
        elif sp<=nb_img and ligne[1]=="spiral":
           X.append(io.imread( image_path + ligne[0] + ".jpg" ))
           Y.append('spiral')
           sp=sp+1
        elif sm>nb_img and sp>nb_img:
            # Quand toutes les images sont enregistrées => sortir de la boucle
            break
        elif sm<=nb_img and sp<=nb_img:
            print(ligne[1] + " inconnu")
            
#print (ligne[0])
X = np.array(X)
Y = np.array(Y)
print(len(Y))

# Conversion de NHWC en NCHW
X = X.transpose(0,3, 1, 2) # La colonne 3 (channel) est délacée entre la 0 (N) et la 1 (H)

#######   3    

# Selectionne aléatoirement n éléments avec le nom name dans le tableau tab
# Retourne un tableau contenant l'index de ces éléments aléatoires
def random_index(tab, name, n):
    index = np.where(tab==name)[0]
    #smooth_index_rand = np.random.choice(smooth_index,10)
    # Avec random.sample : Les élements aléatoires sont bien distincts les uns des autres
    return random.sample(list(index),n) 

# Selection des 10 images aux hazard
smooth_index_rand = random_index(Y,"smooth",10)
spiral_index_rand = random_index(Y,"spiral",10)

# Affichage des spirales
# print ("SPIRAL")
# for i in spiral_index_rand:
#     print(Y[i])
#     plt.imshow(X[i].transpose(1, 2, 0))
#     plt.show()

# Affichage des smooths
# print ("SMOOTH")
# for i in smooth_index_rand:
#     print(Y[i])
#     plt.imshow(X[i].transpose(1, 2, 0))
#     plt.show()

#######   4
# Nombre d'images des données d'entrainements pour chaque classe ici : 0.7*100//2 = 35
nb_train = int(ratio_train * nb_img )

# Récupére la position des images de chaque classe
spiral_index = np.where(Y=="spiral")[0]
smooth_index = np.where(Y=="smooth")[0]    
print(spiral_index)
print(smooth_index)

# Séparation du jeu de données 
train_index = np.concatenate((spiral_index[0:nb_train], smooth_index[0:nb_train]))
test_index = np.concatenate((spiral_index[nb_train:nb_img], smooth_index[nb_train:nb_img]))

X_train = X[train_index].transpose((0,2,3,1)) # Remise au format NHWC
Y_train = Y[train_index]
X_test = X[test_index].transpose((0,2,3,1))
Y_test = Y[test_index]

print(X_train.shape,X_test.shape)

# Vérification du jeu de données
# cpt = 1
# for i in X_train:
#     print(cpt,Y_train[cpt-1])
#     plt.imshow(i)
#     plt.show()
#     cpt = cpt + 1

#######   5   
# Nombre d'images du jeu de données d'entrainement
nb_img_train = int(ratio_train*2*nb_img)



X_train_crop = []
# Rogner toutes les images des données d'entrainements
for img in X_train:
    X_train_crop.append(crop_center(img,crop_size,crop_size))
    
X_train_crop = np.array(X_train_crop)


#######   6        
### COLOR
X_train_crop_color = []

cpt=0
X_train_crop_grey_mean = []
for img in X_train:
    m=center_color(img)
    # i = to_grey(img)
    # m = int(np.mean(i))
    #print(m)
    X_train_crop_grey_mean.append(m)
    # Calcul du seuil optimale de binarisation
    #t = otsu_threshold(X_train_crop_grey[cpt])

cpt = 0
# print("spiral")
# for cpt in range(35):
#     print(X_train_crop_grey_mean[cpt])
# cpt = 0
# print("SMOOTH")
# for cpt in range(35):
#     print(X_train_crop_grey_mean[cpt+35])


X_train_crop_grey_mean = np.array(X_train_crop_grey_mean)

Spiral_mean = np.mean(X_train_crop_grey_mean[0:nb_train//2])
Smooth_mean = np.mean(X_train_crop_grey_mean[nb_train//2:nb_train])
print(Spiral_mean,Smooth_mean)
th = (Spiral_mean + Smooth_mean)/2 # reviens à faire la moyenne de toutes les images
print (th)
# Détermination du seuil
#th = np.median(X_train_crop_grey_mean)
th = otsu_threshold(X_train_crop_grey_mean)
print (th)

classe_test = []
for test in X_train_crop_grey_mean:
    if test>th:
        classe_test.append("SMOOTH")
    else:
        classe_test.append("SPIRAL")

classe_test = np.array(classe_test)
nb_spiral= np.sum(1 * (classe_test[0:35] == "SPIRAL")) 
nb_smooth = np.sum(1 * (classe_test[35:70] == "SMOOTH")) 

print(classe_test[0:35])
print(classe_test[35:70])
print(nb_spiral,nb_smooth)







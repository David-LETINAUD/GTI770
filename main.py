#######   Initialisation
from skimage import io
import numpy as np
import csv
import matplotlib.pyplot as plt
import random


image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"


# Nombre d'images de chaque classe
nb_img = 50

ratio_train = 0.7

crop_size = 50

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

# Rogne de manière centrée l'image img aux dimensions cropx/cropy
# Retourne l'image rognée
def crop_center(img,cropx,cropy):
    x = img.shape[0] # Sauvegarde la taille de l'image en x
    y = img.shape[1]
    startx = x//2-(cropx//2) # '//' Renvoie la partie décimale du quotient.
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

X_train_crop = []
# Rogner toutes les images des données d'entrainements
for img in X_train:
    X_train_crop.append(crop_center(img,crop_size,crop_size))
    
X_train_crop = np.array(X_train_crop)


#######   6        
print("SPIRAL")
plt.imshow(X_train_crop[0])
plt.show()
print("SMOOTH")
plt.imshow(X_train_crop[int(ratio_train*2*nb_img) - 1])
plt.show()

#######   7
# Plusieurs façons de supprimer les 3 canaux de couleurs pour n'en faire qu'1 gris
# On peut par exemple donner un poids différents à chaque couleur
# img : image à griser 
# Retourne une image grise 
def to_grey(img):
    return 0.2989*img[:,:,0] + 0.5870*img[:,:,1] + 0.1140*img[:,:,2]

# Transforme une image grise en image binaire à partir d'un seuil
# Les pixels de valeurs supérieurs seront associés à 1 et ceux à valeurs inférieurs à 0
# img : image à binariser ; seuil : seuil de binarisation
# Retourne l'image binaire
def to_binary(img,seuil): 
    return 1.0 *(img>seuil) # Si la condition est vrai : 1.0*True = 1 sinon 1.0*false = 0    

# Plusieurs manière de définir un seuil de binarisation
# Une méthode très efficasse est l'algorithme d'Otsu
# Il permet de découper l'histogramme en 2 classes tout en minimisant la variance intraclasse (maximiser la variance interclasse)
# Cette fonction provient du site : http://www.degeneratestate.org/posts/2016/Oct/23/image-processing-with-numpy/
# Renvoie le seuil de binarisation calculé
def otsu_threshold(im):
    # Calcul de l'histogramme : 
    # Compte le nombre d'occurences de chaque couleurs (variations de gris)
    pixel_counts = [np.sum(im == i) for i in range(256)]

    # Stocke la variance inter-classe maximale
    s_max = (0,-10)
    # Stocke toutes les variances
    ss = []

    # Parcours tous les seuils possibles
    for threshold in range(256):

        # Mise à jour
        # Découpe l'histogramme en 2 classes en fonction du seuil actuel
        # Calcul de la probabilité de chaque classe
        w_0 = sum(pixel_counts[:threshold])
        w_1 = sum(pixel_counts[threshold:])

        # Calcul des moyennes de chaque classe
        mu_0 = sum([i * pixel_counts[i] for i in range(0,threshold)]) / w_0 if w_0 > 0 else 0       
        mu_1 = sum([i * pixel_counts[i] for i in range(threshold, 256)]) / w_1 if w_1 > 0 else 0

        # Calcul de la variance inter-classe            
        s = w_0 * w_1 * (mu_0 - mu_1) ** 2
        ss.append(s)

        # Si la variance inter-classe maximale a été battu, sauvegarde de celle-ci dans s_max
        if s > s_max[1]:
            s_max = (threshold, s)

    # Retourne le seuil pour lequel la variance inter-classe est maximale (variance intra-classe minimale)
    return s_max[0]


X_train_binary = np.full_like(X_train_crop, 0);
X_train_binary=X_train_binary[:, :, :, 0]

cpt=0
for img in X_train_crop:
    # Conversion en image gris
    X_train_binary[cpt] = to_grey(img)
    # Calcul du seuil optimale de binarisation
    t = otsu_threshold(X_train_binary[cpt])
    # Binarisation de l'image
    X_train_binary[cpt] = to_binary(X_train_binary[cpt],t)
    cpt = cpt + 1

#for img in X_train_binary:   
#    plt.imshow(img, cmap=plt.cm.gray) 
#    plt.show()

#######   8
X_train_plot =np.zeros((X_train_binary.shape[0], 2))

X_train_plot[:,0] = [np.sum(i == 0) for i in X_train_binary]
X_train_plot[:,1] = [np.sum(i == 1) for i in X_train_binary]

####### 9
# On peut s'en convaincre en traçant l'histogramme des 2 classes
# Un seuil pourrait peut-être apparraître pour séparer les 2 classes à partir de la binarisation de l'image 

# Histogramme des 0 ou des 1
ones_or_zeros = 0
# Découper l'intervalle en n_div sous-intervalles 
n_div = 9

# Détermination des intervalles à utiliser pour l'histogramme
min_train_smooth = min(X_train_plot[int(ratio_train*nb_img):int(ratio_train*2*nb_img),ones_or_zeros])
max_train_smooth = max(X_train_plot[int(ratio_train*nb_img):int(ratio_train*2*nb_img),ones_or_zeros])
max_train_smooth=int(round(max_train_smooth/100)*100) # Arrondi à la centaine supérieur
min_train_smooth=int(min_train_smooth/100)*100        # Arrondi à la centaine inférieur

min_train_spiral = min(X_train_plot[:int(ratio_train*nb_img),ones_or_zeros])
max_train_spiral = max(X_train_plot[:int(ratio_train*nb_img),ones_or_zeros])
max_train_spiral=int(round(max_train_spiral/100)*100)
min_train_spiral=int(min_train_spiral/100)*100

range_smooth = range(min_train_smooth,max_train_smooth,(max_train_smooth-min_train_smooth)//n_div)
range_spiral = range(min_train_spiral,max_train_spiral,(max_train_spiral-min_train_spiral)//n_div)

# Histogramme de la classe spiral 
plt.hist(X_train_plot[ :int(ratio_train*nb_img),ones_or_zeros], bins = range_spiral ,histtype='step', stacked=True, fill=False)
# Histogramme de la classe smooth 
plt.hist(X_train_plot[int(ratio_train*nb_img):int(ratio_train*2*nb_img),ones_or_zeros], bins = range_smooth,histtype='step', stacked=True, fill=False)


# Affichage des histogramme
labels= ["Spiral","Smooth"]
plt.legend(labels,loc='upper left')
plt.title('Histogramme')
plt.xlabel("Nombre de 0")  
plt.ylabel("Occurences")

plt.show()

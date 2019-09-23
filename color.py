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

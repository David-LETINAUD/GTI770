`#! /usr/bin/env python3                                                                                                      
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
from zoo_tree import zoo_tree

########################################   Initialisations   ########################################                                                                                
dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_feature_vectors.csv"

# Nombre d'images total du dataset (training + testing)                                                                                                                              
nb_img = 100
# Pourcentage de données utilisées pour l'entrainement                                                                                                                               
ratio_train = 0.7


X=[]
Y=[]

########################################   Lecture   ########################################                                                                                        
# Lecture du fichier CSV                                                                                                                                                             
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                        
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        #print(type(features),type(galaxy_class))                                                                                                                                    

        X.append(features)
        Y.append(galaxy_class)


############## FIN LECTURE #########################



# Imports                                                                                                                                                 
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler

#pour les premiers tests                                                                                                                                  
from sklearn import datasets
iris = datasets.load_iris()

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target)


#scale data + multinomial bayes                                                                                                                           
scaler = MinMaxScaler()
#print(scaler.fit(data))                                                                                                                                  
scale_data = scaler.fit(iris.data, iris.target).predict(iris.data)
mnb = MultinomialNB()
print(mnb.fit(iris.data,iris.target))

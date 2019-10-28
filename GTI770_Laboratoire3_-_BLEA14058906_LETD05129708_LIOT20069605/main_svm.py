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
from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier# Import Decision Tree Classifie
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
sns.set
import sklearn.metrics as metrics

import csv
import matplotlib.pyplot as plt

import operator
import numpy as np 
import random
import math
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.svm import SVC

from sklearn.metrics import classification_report , confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

########################################   Initialisations   ########################################

#image_path = "C:/Users/David/Desktop/GTI770/data/data/images/"
#image_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/images/'
#dataset_path = "C:/Users/David/Desktop/GTI770/data/data/csv/galaxy/galaxy_label_data_set.csv"
#dataset_path = '/Users/thomas/Desktop/COURS_ETS/gti770/data/csv/galaxy/galaxy_label_data_set.csv'
dataset_path = "/home/alex/Desktop/GTI770-tp2/csv/galaxy/galaxy_feature_vectors.csv"
image_path = "/home/alex/Desktop/GTI770-tp2/csv/images/"
mail_data_path="/home/alex/Desktop/GTI770-tp2/csv/spam/spam.csv"
# separt la matrice de date en 2 matrice
#on doit avoir au moin 2 matrice vide créer avant l'appel de la méthode

#def SplitData(Originaldata,ratio,XTrain=[], Xtest=[]):
 #   for x in range(Originaldata):
  #      if random.random()< ratio:
   #         XTrain.append(x)
    #    else:
     #       Xtest.append(x)
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
########################################   Separation galaxy  ######################################## 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train, random_state=1) # 70% training and 30% test

########################################   fin separation   ######################################## 
nb_mail=1000
X_mail=[]
Y_mail=[]  
    
########################################   Lecture Spam   ######################################## 
with open(mail_data_path, 'r') as f:
    mail_features_list = list(csv.reader(f, delimiter=','))


    # Lecture ligne par ligne                                                                                                                                                        
    for c in range(nb_mail):
        mail_features = [float(i) for i in mail_features_list[0][0:57]]
        mail_class = int(float( mail_features_list[0][57]))
        mail_features_list.pop(0)
                                                                                                                                   

        X_mail.append(mail_features)
        Y_mail.append(mail_class)
        #print(X_mail)
        #print("--------------Ymail--------------")
        #print( Y_mail)
        
############## FIN LECTURE SPAM #########################            
        
########################################   Separation mail   ######################################## 
X_mail_train, X_mail_test, Y_mail_train, Y_mail_test = train_test_split(X_mail, Y_mail, train_size=ratio_train, random_state=1) # 70% training and 30% test

############## FIN separation mail#########################  

C_tab=[0.001,0.1,1,10]
gamma=[0.001,0.1,1,10]
#kernel=['linear','rbf']
kernel="linear"
Kernel2="rbf"
def SVCLine(X_train,Y_train,X_test,Y_test,c):
    

    svc_class = svm.SVC(kernel="linear",C=c)
    svc_class.fit(X_train,Y_train)
    y_pred= svc_class.predict(X_test)

    print(confusion_matrix(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))
    
def SVC_rbf(X_train,Y_train,X_test,Y_test,c,gamma):
    svc_class = svm.SVC(kernel="rbf",C=c,gamma=gamma)
    svc_class.fit(X_train,Y_train)
    y_pred= svc_class.predict(X_test)

    print(confusion_matrix(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))
    


for C in C_tab:
    print("Kernel Type","linear","valeur c", C)
    SVCLine(X_train,Y_train,X_test,Y_test,C)    

for gamma in gamma:
   
    for C in C_tab:
        print("Kernel Type rbf","valeur c", C, "valeur gamma",gamma)
        SVC_rbf(X_train,Y_train,X_test,Y_test,c,gamma)        



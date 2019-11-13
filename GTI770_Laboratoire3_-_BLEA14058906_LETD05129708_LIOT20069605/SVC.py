# %%
# ! /usr/bin/env python3
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
Dl pycharm sur linux
"""
# from skimage import io
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifie
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

sns.set
import sklearn.metrics as metrics
# from sklearn.metric import accuracy_score, f1_score
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# from color import center_color,crop_center
# from fourier_transform import fourier_transform
# from binaryPattern import binaryPatterns
# import match
import operator
import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit,GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.svm import SVC
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import pandas as pd


########################################   Initialisations   ########################################

####path ordi ETS
dataset_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/galaxy/galaxy_feature_vectors.csv"

image_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/images/"

mail_data_path = "/home/ens/AN03460/Desktop/Gti-770/First tp3/data/data/csv/spam/spam.csv"

### path ordi Alex
#dataset_path = "/home/alex/Desktop/lab3/GTI770-AlexandreBleau_TP3-branch/GTI770_Laboratoire3_-_BLEA14058906_LETD05129708_LIOT20069605/data/csv/galaxy/galaxy_feature_vectors.csv"
#dataset_path = "/home/alex/Desktop/lab3/GTI770-AlexandreBleau_TP3-branch/GTI770_Laboratoire3_-_BLEA14058906_LETD05129708_LIOT20069605/data/csv/galaxy/TP1_features.csv"
nb_img = 160

# Pourcentage de données utilisées pour l'entrainement
ratio_train = 0.7

X = []
Y = []

########################################   Lecture   ########################################
# Lecture du fichier CSV
with open(dataset_path, 'r') as f:
    features_list = list(csv.reader(f, delimiter=','))

    # Lecture ligne par ligne
    for c in range(nb_img):
        features = [float(i) for i in features_list[0][1:75]]
        galaxy_class = int(float(features_list[0][75]))
        features_list.pop(0)
        # print(type(features),type(galaxy_class))

        X.append(features)
        Y.append(galaxy_class)

############## FIN LECTURE #########################
########################################   Separation galaxy  ########################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio_train,
                                                    random_state=1)  # 70% training and 30% test

########################################   fin separation   ########################################


#
from sklearn import svm

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

#pyt    % matplotlib inline
C = [0.001, 0.1, 1, 10]
gamma = [0.001, 0.1, 1.0, 10.0]
# kernel=['linear','rbf']

#def Stratified(n_split,size,radom):
 #   # faire un  split # test a 20 %
  #  Split = StratifiedShuffleSplit(n_split=n_split,test_size=size,random_state=random)
   # return Split

#utriliser plus tard de la faacon suivant
# valeur de retour.split(X,Y)
#fair une for pour chaque element si on veut les utiliser


#finir les modification pe fair eune 2e, methode pour rdf.

def GridSearch_bestparam(X_train,Y_train):
    print('ca commence')

    param = [{'C':[0.001,0.1,1,10],'kernel':['linear']},
             {'C': [0.001,0.1,1,10],'gamma':[0.001, 0.1,1,10], 'kernel': ['rbf']}, ]
    #param = {'kernel':("linear","rbf"), 'C':[0.001,0.1,1,10]}
    ##param= {'C': [0.001, 0.1, 1],'gamma':[0.001, 0.1, 1 ], 'kernel': ['rbf']}
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
   # param = [{'kernel': ("rbf"), 'gamma': [0.001, 0.1, 1, 10], 'C': [0.001, 0.1, 1, 10]},{'kernel': ( "linear"),'C': [0.001, 0.1, 1, 10]}]
    svc =svm.SVC(gamma= "scale",cache_size=11264)
  #  clf = GridSearchCV(svc,param,scoring=scoring, cv=5,refit = 'AUC', return_train_score=True)
    clf = GridSearchCV(svc, param, scoring=scoring, cv=5, refit='AUC', return_train_score=True,n_jobs=10)
    clf.fit(X_train,Y_train)

    #print("value")
    #print(clf.cv_results_.keys())

    #return clf
    #print('ca fini')
    #pandas dataframe
    #print('grid test')
    print('best param')
    print(clf.best_params_)
    print('best score')
    print(clf.best_score_)


    return clf
    #return clf.cv_results_

    #return sorted(clf.cv_result_.key())
        



def SVCLine(X_train, Y_train, X_test, Y_test,C):

    svc_class = svm.SVC(kernel="linear", C=C)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))


def SVC_rbf(X_train, Y_train, X_test, Y_test,C,gamma):
    svc_class = svm.SVC(kernel="rbf", C=C, gamma=gamma)
    svc_class.fit(X_train, Y_train)
    y_pred = svc_class.predict(X_test)

    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
# print('Linear')
# SVCLine(X_train, Y_train, X_test, Y_test,10)
# print('rbf')
# SVC_rbf(X_train, Y_train, X_test, Y_test,1,1)

# Grid = GridSearch_bestparam(X_train,Y_train)
# #print(Grid)
# result = Grid.cv_results_
#
# df = pd.DataFrame(data=result)
#
# df[['param_kernel','param_C','param_gamma','mean_train_Accuracy','mean_fit_time','mean_score_time']]
# print(df)
# #print(df.get_values(1,'mean_train_Accuracy'))
# list_accuracy=[]
# list_time=[]
# list_Param_C=[]
# list_gamma=[]
# list_kernel=[]
# list_test_acc=[]
# list_std_train_acc=[]

# for i in range(15):
#     list_accuracy.append(df.get_value(i,35,'mean_train_Accuracy'))
#     list_time.append(df.get_value(i,0,'mean_fit_time'))
#     list_Param_C.append(df.get_value(i,4,'param_C'))
#     list_gamma.append(df.get_value(i,6,'param_gamma'))
#     list_kernel.append(df.get_value(i,5,'param_kernel'))
#     list_test_acc.append(df.get_value(i,28,'mean_test_Accuracy'))
#     list_std_train_acc.append(df.get_value(i, 36, 'std_train_Accuracy'))
#
#
#
# print(list_accuracy)
# print(list_time)
# print(list_Param_C)
# print(list_gamma)
# print(list_kernel)
# print(list_test_acc)
# print(list_std_train_acc)

# print(Grid.cv_results_)
# print(Grid)
# print(type(Grid.cv_results_))



#Grid_1(Grid,mean_train_Accuracy ,mean_fit_time,"estimator","Param 2 test")




# plt.plot(list_Param_C[0:5],list_accuracy[0:5],'x',label = "Param C linear" )
#
# plt.xlabel('Param C')
# plt.ylabel('accuracy')
# plt.title('Meilleur accuracy en fonction de C Linear')
#
#
#
#
#
#
# plt.legend()
#
# plt.show()
#
# plt.plot(list_time[0:5],list_Param_C[0:5],'x',label = "Temps linear " )
# plt.xlabel('temps')
# plt.ylabel('accuracy')
# plt.title('Meilleur accuracy en fonction du temps de calcule Linear')
# plt.legend()
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
# x_line,y_line = np.meshgrid(list_gamma[5:19],list_Param_C[5:19])
# z_line = np.tile(list_accuracy[5:19],(len(list_accuracy[5:19]),1))
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.set_xlabel('param Gamma')
# ax.set_ylabel('Param C')
# ax.set_zlabel('Accuracy')
#
# ax.plot_surface(x_line,y_line,z_line,cmap='ocean')
# ax.set_title('Accurace en fonction de C et gamma ')
#
# plt.show()
#
#

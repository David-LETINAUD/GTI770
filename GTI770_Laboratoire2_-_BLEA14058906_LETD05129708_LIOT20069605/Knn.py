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
Dl pycharm sur linux
"""
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#sns.set
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler


 def accknn(matrice):
  """
  Fonction qui permet de calculer la precision de notre matrice de confusion

  input :
      matrice (nparray) : matrice de confusion

  output : 
      Precision(int)    : suivant le calcul (Tp+Fp)/(Tp+fp+fn+tn)

  """
    deno = matrice[0][0] + matrice[1][1]

    nume = matrice[0][0] + matrice[0][1] + matrice[1][0] + matrice[1][1]

    acc = (float(deno) / float(nume))
    return acc


def KNN(Xtrain, Xtest, Ytrain, Ytest, k):
 """
 Fonction qui permet de calculer la precision et la valeur du F1_score selon un parametre K voisin

 input :
     XTrain (nparray):  tableau des features destiné à l'entrainement.
     Xtest  (nparray):  tableau des features destiné aux tests.
     Ytrain (nparray):  tableau des étiquettes associée aux valeurs d'entrainement.
     Ytest  (nparray):  tableau des étiquettes pour les valeurs de test.
     k (int):           valeur numerique K pour determiner le nombre de voisin interogé

 output : 
     Acc (int)       : précision selon la valeur k 
     score_(int)     : F1_score selon la valeur k

 """
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)  # scale des data entre 0 et 1 par défaut.
    X_train_scale = scaler.fit_transform(Xtrain)  # On scale les data d'entrainement
    X_test_scale = scaler.fit_transform(Xtest)  # On scale les data de test

    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train_scale, Ytrain)
    y_pred = knn.predict(X_test_scale)
    matrice = confusion_matrix(Ytest, y_pred)

    acc_ = accknn(matrice)
    score_ = metrics.f1_score(Ytest, y_pred, labels=None, pos_label=1, average="weighted", sample_weight=None)

    return ([acc_, score_])

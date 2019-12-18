"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 4 — Développement d’un système intelligent

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight
import time
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


def PCA_Find_ncomp(X_train, Variance):
    """
        Fonction De PCA qui retourne le nombre de composant a utiliser pour obtenir la variance désirée
        input :
            X_train : (np_array) Liste des vecteurs à analysé pour l'entrainement
            Variance: (double) Entre 1 et 0 représente la variance désirée

        output:
              (int) Nombre de composent pour atteindre la variance désiré

    """

    # pca = PCA(n_components=20)
    pca = PCA(Variance)
    # LogReg= LogisticRegression(multi_class='auto',solver='liblinear')
    pca.fit(X_train)
    return pca.n_components_


def PCA_transform(X_train, X_test, N_comp):
    """
           Fonction De PCA qui retourne le array transformer selon le nombre de composant N_comp
           input :
               X_train : (np_array) Liste des vecteurs à analysé pour l'entrainement
               X_test  : (np_array) Liste des vecteurs à analysé pour les tests
               N_comp  : (int) Nombre de composant

           output:
               PCA_X_train: (np.array) Liste des vecteurs d'entaînement transformé pas PCA
               PCA_X_test : (np.array) Liste des vecteurs de test transformé pas PC

       """
    pca = PCA(n_components=N_comp)
    pca.fit(X_train)
    PCA_X_train = pca.transform(X_train)
    PCA_X_test = pca.transform(X_test)
    return PCA_X_train, PCA_X_test


def SVM_Gridsearch(X_train, Y_train):
    """
        Fonction GridSearchCv qui permet de trouver les meilleurs hyperparamètres
        input :
            X_train:(np_array) Liste des vecteurs à analysé
            Y_train:(np_array)liste de la classification des vecteurs
        output:
            Grid:(sklearn.model_selection._search.GridSearchCV) Résultat de la fonction gridsearch
    """
    param = [{'C': [0.001, 0.1, 1, 10], 'kernel': ['linear']},
             {'C': [0.001, 0.1, 1, 10], 'gamma': [0.001, 0.1, 1, 10], 'kernel': ['rbf']}, ]

    acc = make_scorer(accuracy_score)
    f1 = make_scorer(f1_score, average='macro')

    score = {'F1': f1, 'Accuracy': acc}
    #score = {'Accuracy': acc}

    svc = svm.SVC(gamma="scale", cache_size=11264)

    clf = GridSearchCV(svc, param, scoring=score, cv=5, refit='Accuracy', return_train_score=True, n_jobs=10)
    clf.fit(X_train, Y_train)

    return clf


def SVC_Linear(X_train, Y_train, X_test, Y_test, C):
    """
        Fonction svc linear qui calcule la matrice de confusion selon l'hyperparamètre choisi
        input :
            X_train: (np_array) Liste des vecteurs à analysé pour l'entrainement
            Y_train: (np_array) liste de la classification des vecteurs pour l'entrainement
            X_test : (np_array) Liste des vecteurs pour à analyser pour le test
            Y_test : (np_array) liste de la classification des vecteurs pour le test
            C      : (double) hyperparamètre C
         output:
            Matrice de confusion
            Rapport de classification

    """
    svc_class = svm.SVC(kernel="linear", C=C)
    start_train = time.time()
    svc_class.fit(X_train, Y_train)
    end_train = time.time()
    start_pred = time.time()
    y_pred = svc_class.predict(X_test)
    end_pred = time.time()

    train_time = (end_train - start_train)
    pred_time = (end_pred - start_pred)
    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    print('Temps de training:', train_time, 'Temps de prédiction: ', pred_time)
    return confusion_matrix(Y_test, y_pred), classification_report(Y_test, y_pred), train_time, pred_time


def SVC_rbf(X_train, Y_train, X_test, Y_test, C, gamma):
    """
        Fonction svc RBF qui calcule la matrice de confusion selon les hyperparamètres choisis
        input :
            X_train: (np_array) Liste des vecteurs à analysé pour l'entrainement
            Y_train: (np_array) liste de la classification des vecteurs pour l'entrainement
            X_test : (np_array) Liste des vecteurs pour à analyser pour le test
            Y_test : (np_array) liste de la classification des vecteurs pour le test
            C      : (double) hyperparamètre C
            gamma  : (double)hyperparamètre gamma
        output:
             svc_class: Resultat
             y_pred: (np_array) Liste des prédiction
	     train_time: (int) temps de training
             pred_time: (int) temps de prédiction

    """
    svc_class = svm.SVC(kernel="rbf", C=C, gamma=gamma,cache_size=11264,probability=True,class_weight='balanced')

    start_train = time.time()
    svc_class.fit(X_train, Y_train)
    end_train = time.time()
    start_pred = time.time()
    y_pred = svc_class.predict(X_test)
    end_pred = time.time()

    train_time = (end_train - start_train)
    pred_time = (end_pred - start_pred)


    print(confusion_matrix(Y_test, y_pred))
    print(classification_report(Y_test, y_pred))
    print('Temps de training:', train_time, 'Temps de prédiction: ', pred_time)
    #return confusion_matrix(Y_test, y_pred), classification_report(Y_test, y_pred), train_time, pred_time
    return svc_class,y_pred,train_time, pred_time



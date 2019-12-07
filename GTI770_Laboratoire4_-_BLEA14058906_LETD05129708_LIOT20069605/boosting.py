#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 4 - Développement d’un système intelligent

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

# inspiré de : https://www.python-course.eu/Boosting.php
from functions import *
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from RN_model import RN_model
import pickle
from sklearn.decomposition import PCA
from SVC_model import PCA_Find_ncomp

from sklearn import metrics

layer_sizes = [500] # OK
epochs = 100 # OK avec 100
learning_rate = 0.001
batch_size = 500

dropout = 0.5



def boosting(data_path, weights, RN_path, RF_path, SVM_path ):
    X, Y, id, le = get_data(data_path)
    X = preprocessing.normalize(X, norm='max',axis = 0)

    # PCA pour SVM
    N_comp=PCA_Find_ncomp(X,0.95)

    pca = PCA(n_components=32)
    pca.fit(X)
    PCA_X = pca.transform(X)

    nb_features = len(X[0])
    nb_classes = max(Y)+1

    #_, X_test, id_train, id_test, _, Y_test = train_test_ID_split(X,Y, id)

    # LOAD modeles
    RN_model_ = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
    RN_model_.load_weights(RN_path)

    # pickle_in = open(RF_path, "rb")
    # RF_model_ = pickle.load(pickle_in)

    pickle_in = open(SVM_path, "rb")
    SVM_model_ = pickle.load(pickle_in)


    #########################   predict modele
    # RN MODEL
    #Y_pred_RN = RN_model_.predict_proba(X)

    # RF MODEL
   # Y_pred_RF = RF_model_.predict_proba(X)

    #SVM MODEL
    Y_pred_SVM = SVM_model_.predict(PCA_X[:10])
    print("###################")
    print(Y_pred_SVM)
    print("###################")

    Y_pred_one_hot = Y_pred_SVM# weights[0] * Y_pred_RN + weights[2]*Y_pred_SVM # weights[1] * Y_pred_RF + weights[2]*Y_pred_SVM 
    
    Y_pred = []
    for i in Y_pred_one_hot:
        Y_pred.append(np.argmax(i))

    Y_pred_label = list(le.inverse_transform(Y_pred))
    
    # return id/Y_pred/acc/f1
    f1 = metrics.f1_score(Y, Y_pred,average='weighted')
    acc = metrics.accuracy_score(Y, Y_pred)
    print("acc :", acc,"f1 :", f1)
    return id, Y_pred_label, acc, f1


def run_boosting(data_path_tab, weights_tab, RN_path, RF_path, SVM_path):
   
    id_genre_pred = []
    perf = []
    for data_path,weights,rn_p,rf_p,svm_p in zip(data_path_tab,weights_tab, RN_path, RF_path, SVM_path):
        r = boosting(data_path,weights,rn_p,rf_p,svm_p)
        id_genre_pred.append(r[0:2])
        perf.append(r[2:4])

    return id_genre_pred, perf


data_path = ["./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv", "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv", "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"] #=> MLP 30.7%
# Calculer les poids
#           RN    RF  SVM    
MSSD_acc = [0.353, 0.2786, 0.31]
MFCC_acc = [0.249,0.2842,0.07]
MARSYAS_acc = [0.32,0.2624,0.27]

# RN_acc = [0.353,0.249, 0.320]
# RN_f1 = [0.333,0.220,0.299]
weight = []
# Le poids est calculé selon le pourcentage que représente l'accuracy..
# .. du modèle sur la somme total des accuracy sur le dataset étudié
MSSD_total = np.sum(np.array(MSSD_acc))
weight.append([a/MSSD_total for a in MSSD_acc])
MFCC_total = np.sum(np.array(MFCC_acc))
weight.append([a/MFCC_total for a in MFCC_acc])
MARSYAS_total = np.sum(np.array(MARSYAS_acc))
weight.append([a/MARSYAS_total for a in MARSYAS_acc])
print(weight)


RN_models_path = ["Models/MLP_model_SSD/cp.ckpt", "Models/MLP_model_MFCC/cp.ckpt", "Models/MLP_model_MARSYAS/cp.ckpt" ]
RF_models_path = ["./Models/rfc_ssd.sav","./Models/rfc_mfcc.sav","./Models/rfc_marsyas.sav"]
SVM_models_path = ['./Models/svm_ssd.sav',"./Models/svm_mfcc.sav","./Models/svm_marsyas.sav"]

run_boosting(data_path,weight,RN_models_path, RF_models_path, SVM_models_path)

# # LOAD modeles
# RN_model_ = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
# RN_model_.load_weights("Models/MLP_model_SSD/cp.ckpt")

# Y_pred_RN = RN_model_.predict(X_test)



##### MARCHE BIEN
# dp = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"
# X, Y, id, le = get_data(dp)
# X = preprocessing.normalize(X, norm='max',axis = 0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)
# X_train_v2, X_test_v2, id_train_v2, id_test_v2, Y_train_v2, Y_test_v2 = train_test_ID_split(X,Y, id)

# nb_features = len(X[0])
# nb_classes = max(Y)
# train_size = len(X)

# model2 = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
# model2.load_weights("Models/MLP_model_SSD/cp.ckpt")

# Y_pred_temp2 = model2.predict(X_test_v2)

# remise en forme de Y_pred
# Y_pred2 = []
# for i in Y_pred_temp2:
#     Y_pred2.append(np.argmax(i))    

# f1 = metrics.f1_score(Y_test, Y_pred2,average='weighted')
# acc = metrics.accuracy_score(Y_test, Y_pred2)
# print("acc :", acc,"f1 :", f1)

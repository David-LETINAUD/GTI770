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

def run_combining(data_path_tab, weights, RN_path, RF_path, SVM_path,SVM_N_comp, with_labels = True):
    X_list = []
    Y_list = []

    if with_labels == True:
        for data_path in data_path_tab:
            gd = get_data(data_path)
            X_list.append(gd[0])
            Y_list.append(gd[1])
        id = gd[2]
        le = gd[3]
    else :
        for data_path in data_path_tab:
            gd = get_data_whithout_labels(data_path)
            X_list.append(gd[0])
        id = gd[1]
        classes_ = gd[2]


    # dataset_size = 1000 #len(X)
    # X_list = X_list[:,:dataset_size]
    # if with_labels == True:
    #     Y_list = Y_list[:][:dataset_size]


    X_list_normalize = []
    for X in X_list:
        X_list_normalize.append(preprocessing.normalize(X, norm='max',axis = 0))
    X_list = X_list_normalize


    # PCA pour SVM (si SVM_N_comp<0 => PCA non utilisé )
    if SVM_N_comp>0:
        pca = PCA(n_components=SVM_N_comp)
        pca.fit(X_list[2])
        PCA_X = pca.transform(X_list[2])
    else:
        PCA_X = X_list[2]

    # Calcul nb de features et de classes
    nb_features = len(X_list[0][0])
    if with_labels == True:
        nb_classes = max(Y_list[0])+1
    else :
        nb_classes = len(classes_)

    # LOAD modeles
    RN_model_ = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
    RN_model_.load_weights(RN_path)

    pickle_in = open(RF_path, "rb")
    RF_model_ = pickle.load(pickle_in)

    pickle_in = open(SVM_path, "rb")
    SVM_model_ = pickle.load(pickle_in)

    #########################   predict modele
    # RN MODEL
    Y_pred_RN = RN_model_.predict_proba(X_list[0])

    # RF MODEL
    Y_pred_RF = RF_model_.predict_proba(X_list[1])

    #SVM MODEL
    Y_pred_SVM = SVM_model_.predict_proba(PCA_X)

    ######################### Combinaison des décisions
    #Y_pred_one_hot = Y_pred_SVM
    Y_pred_one_hot = weights[0] * Y_pred_RN + weights[1] * Y_pred_RF + weights[2]*Y_pred_SVM 

    Y_pred = []
    for i in Y_pred_one_hot:
        Y_pred.append(np.argmax(i))

    Y_pred_label = [classes_[i] for i in Y_pred ]

    return id, Y_pred_label


data_path = ["./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv", "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv", "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"] #=> MLP 30.7%
#data_path_nolabels = ["./untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv", "./untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv", "./untagged_feature_sets/msd-marsyas_test_new_nolabels/msd-marsyas_test_new_nolabels.csv"] #=> MLP 30.7%
#data_path_nolabels = [ "./untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv",  "./untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv", "./untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv"] #=> MLP 30.7%
data_path_nolabels = ["./untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv","./untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv","./untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv"]
# Calculer les poids
#           RN    RF  SVM    
# MSSD_acc = [0.315, 0.2786, 0.31]
# MFCC_acc = [0.249,0.2842,0.26]
# MARSYAS_acc = [0.31,0.2624,0.27]

Acc = [0.155, 0.13, 0.117]

sum_acc = np.sum(Acc)
weight = [a/sum_acc for a in Acc]
#weight = [0.55, 0.1 , 0.35]
print(weight)

RN_models_path = "Models/MLP_model_MFCC/cp.ckpt"
#RF_models_path = "./Models/rfc_mfcc.sav"
RF_models_path = "./Models/rfc_mfcc.sav"
#SVM_models_path = "./Models/svm_marsyas.sav"
SVM_models_path = "./Models/svm_mfcc.sav"
SVM_N_comp= -1

#run_combining(data_path,weight,RN_models_path, RF_models_path, SVM_models_path,SVM_N_comp_tab,with_labels=True)

pred = run_combining(data_path_nolabels,weight,RN_models_path, RF_models_path, SVM_models_path,SVM_N_comp,with_labels=False)


def write_pred_csv(title_csv,prediction_list):
    
    """
        ecriture csv, ! format prediction_list,
        """
    
    with open(title_csv, mode = 'w') as pred_file :
        file_writer = csv.writer(pred_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['id','genre'])
        for i in range(len(prediction_list[0])):
            file_writer.writerow([prediction_list[0][i],prediction_list[1][i]])


write_pred_csv("3_model_MFCC.csv",pred)
# write_pred_csv(list_files[1],pred_Y)
# write_pred_csv(list_files[2],pred_Z)


#Final SSD_acc = 0.26981 -> 0.404
#Final MFCC_acc = 0.15386 -> 0.231
#Final Marsyas_acc = 0.24379 -> 0.365
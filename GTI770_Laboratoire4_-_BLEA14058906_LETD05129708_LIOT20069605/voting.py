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

# Hyperparamètre des réseaux de neurones
layer_sizes = [500]
epochs = 100 
learning_rate = 0.001
batch_size = 500

dropout = 0.5


def voting_L1(data_path, weights, RN_path, RF_path, SVM_path, SVM_N_comp,classes_, with_labels = True ):
    if with_labels == True:
        X, Y, id, le = get_data(data_path)
    else :
        X, id = get_data_whithout_labels(data_path)
        Y = [-1]

    # dataset_size = 1000 #len(X)
    # X = X[:dataset_size]
    # if with_labels == True:
    #     Y = Y[:dataset_size]

    X = preprocessing.normalize(X, norm='max',axis = 0)

    # PCA pour SVM (si SVM_N_comp<0 => PCA non utilisé )
    if SVM_N_comp>0:
        pca = PCA(n_components=SVM_N_comp)
        pca.fit(X)
        PCA_X = pca.transform(X)
    else:
        PCA_X = X

    # Calcul nb de features et de classes
    nb_features = len(X[0])
    nb_classes = len(classes_)

    # LOAD modeles
    RN_model_ = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
    RN_model_.load_weights(RN_path)

    pickle_in = open(RF_path, "rb")
    RF_model_ = pickle.load(pickle_in)

    pickle_in = open(SVM_path, "rb")
    SVM_model_ = pickle.load(pickle_in)

    #########################   prediction
    # RN MODEL
    Y_pred_RN = RN_model_.predict_proba(X)

    # RF MODEL
    Y_pred_RF = RF_model_.predict_proba(X)

    #SVM MODEL
    Y_pred_SVM = SVM_model_.predict_proba(PCA_X)

    ######################### Combinaison des décisions
    Y_pred = weights[0] * Y_pred_RN + weights[1] * Y_pred_RF + weights[2]*Y_pred_SVM 
    
    return Y_pred, id, Y
    


def voting(data_path_tab, weights_tab, RN_path, RF_path, SVM_path,SVM_N_comp_tab, with_labels = True):
    # liste des classes tel que le ressort le.classes_
    classes_ = ['BIG_BAND','BLUES_CONTEMPORARY','COUNTRY_TRADITIONAL','DANCE',
                'ELECTRONICA','EXPERIMENTAL','FOLK_INTERNATIONAL','GOSPEL','GRUNGE_EMO',
                'HIP_HOP_RAP','JAZZ_CLASSIC','METAL_ALTERNATIVE','METAL_DEATH',
                'METAL_HEAVY','POP_CONTEMPORARY','POP_INDIE','POP_LATIN','PUNK','REGGAE',
                'RNB_SOUL','ROCK_ALTERNATIVE','ROCK_COLLEGE','ROCK_CONTEMPORARY',
                'ROCK_HARD','ROCK_NEO_PSYCHEDELIA']

        # SSD MFCC MARSYAS
    # ACC = [0.274,0.155, 0.238]        # Accuracy correspondant à la combinaison de 3 modèles RN RF et SVM par features set
    Ponderation = [0.666,0.402,0.599]   # Somme des accuracys de chaque modèles en réponse au dataset SSD MFCC MARSYAS
    ACC = [0.274,0, 0.238]                
    weight_L2 = [p*a for a,p in zip(ACC,Ponderation)]
    sum_L2 = np.sum(np.array(weight_L2))
    weight_L2_normalize = [w/sum_L2 for w in weight_L2]

    Y_pred_proba_tab=[]

    for data_path,weights,rn_p,rf_p,svm_p, n_comp in zip(data_path_tab,weights_tab, RN_path, RF_path, SVM_path,SVM_N_comp_tab):
        r,id, Y = voting_L1(data_path,weights,rn_p,rf_p,svm_p,n_comp,classes_, with_labels)
        Y_pred_proba_tab.append(r)

    #Y_pred_one_hot = weight_L2[0]*Y_pred_one_hot_tab[0] + weight_L2[1]*Y_pred_one_hot_tab[1]+ weight_L2[2]*Y_pred_one_hot_tab[2]
    #Y_pred_one_hot = weight_L2_normalize[0]*Y_pred_one_hot_tab[0] + weight_L2_normalize[1]*Y_pred_one_hot_tab[1] + weight_L2_normalize[2]*Y_pred_one_hot_tab[2]
    Y_pred_proba = weight_L2_normalize[0]*Y_pred_proba_tab[0] + weight_L2_normalize[2]*Y_pred_proba_tab[2]
    Y_pred = []

    # Prendre le vote maximal
    for i in Y_pred_proba:
        Y_pred.append(np.argmax(i))

    # Conversion du nombre en label
    Y_pred_label = [classes_[i] for i in Y_pred ]

    if with_labels == True:
        f1 = metrics.f1_score(Y, Y_pred,average='weighted')
        acc = metrics.accuracy_score(Y, Y_pred)
        return [id, Y_pred_label], [acc,f1]
    else :
         return id, Y_pred_label


data_path = ["./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv", "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv", "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"] #=> MLP 30.7%
data_path_nolabels = ["./untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv", "./untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv", "./untagged_feature_sets/msd-marsyas_test_new_nolabels/msd-marsyas_test_new_nolabels.csv"] #=> MLP 30.7%

# Calculer les poids
#           RN    RF  SVM    
MSSD_acc = [0.273, 0.21, 0.183]
MFCC_acc = [0.155,0.13,0.117]
MARSYAS_acc = [0.224,0.208,0.167]

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
#weight = [[0.4,0.2,0.4], [0.4,0.55,0.05], [0.35,0.3,0.35]]


RN_models_path = ["Models/MLP_model_SSD/cp.ckpt", "Models/MLP_model_MFCC/cp.ckpt", "Models/MLP_model_MARSYAS/cp.ckpt" ]
RF_models_path = ["./Models/rfc_ssd.sav","./Models/rfc_mfcc.sav","./Models/rfc_marsyas.sav"]
SVM_models_path = ["./Models/svm_ssd.sav","./Models/svm_mfcc.sav","./Models/svm_marsyas.sav"]
SVM_N_comp_tab = [28, -1,32]

# Dataset d'entrainement avec label
#pred, perf = voting(data_path,weight,RN_models_path, RF_models_path, SVM_models_path,SVM_N_comp_tab,with_labels=True)

# Dataset de validation sans label
pred = voting(data_path_nolabels,weight,RN_models_path, RF_models_path, SVM_models_path,SVM_N_comp_tab,with_labels=False)

def write_pred_csv(title_csv,prediction_list):
    
    """
        ecriture csv, ! format prediction_list,
        """
    
    with open(title_csv, mode = 'w') as pred_file :
        file_writer = csv.writer(pred_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['id','genre'])
        for i in range(len(prediction_list[0])):
            file_writer.writerow([prediction_list[0][i],prediction_list[1][i]])


write_pred_csv("6_modeles.csv",pred)


#Final SSD_acc = 0.26981 -> 0.404
#Final MFCC_acc = 0.15386 -> 0.231
#Final Marsyas_acc = 0.24379 -> 0.365
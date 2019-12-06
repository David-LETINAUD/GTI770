#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Course :
        
GTI770 — Systèmes intelligents et apprentissage machine

Project :

Lab # 4 — Développement d'un système intelligent

Students :

Alexendre Bleau — BLEA14058906

David Létinaud  — LETD05129708                                                                                                                                                                                     
Thomas Lioret   — LIOT20069605

Group :

GTI770-A19-01
"""

#Imports                                                                                                                                                                            
from functions import *
from RF_model import *
from boosting import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics
import time


#Ouverture
path_SD = "./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv"   #dataset SpectralDerivate
path_MFC = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv"        #dataset MFC
path_SSD = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"                    #dataset SSD
path_jmirderivatives = "./tagged_feature_sets/msd-jmirderivatives_dev/msd-jmirderivatives_dev.csv"
path_jmirlpc = "./tagged_feature_sets/msd-jmirlpc_dev/msd-jmirlpc_dev.csv"                                                                         
path_jmirmoments = "./tagged_feature_sets/msd-jmirmoments_dev/msd-jmirmoments_dev.csv"                                                                                        
path_marsyas = "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"
path_mvd = "./tagged_feature_sets/msd-mvd_dev/msd-mvd_dev.csv"                                                                                      
path_rh = "./tagged_feature_sets/msd-rh_dev_new/msd-rh_dev_new.csv"
path_trh = "./tagged_feature_sets/msd-trh_dev/msd-trh_dev.csv"



data_path = [path_SSD,path_MFC,path_marsyas]
# Calculer les poids
#           SSD    MFCC  MARSYAS
MSSD_acc = [0.353, 0.25, 0.28]
MFCC_acc = [0.249,0.27,0.17]
MARSYAS_acc = [0.32,0.18,0.25]

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
SVM_models_path = ["./Models/rfc_ssd.sav","./Models/rfc_mfcc.sav","./Models/rfc_marsyas.sav"]

run_boosting(data_path,weight,RN_models_path, RF_models_path, SVM_models_path)





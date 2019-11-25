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



L = [[1,2,3,4],[5,6,7,8]]
L = np.array(L)
yo = L[:,1]

print(yo)





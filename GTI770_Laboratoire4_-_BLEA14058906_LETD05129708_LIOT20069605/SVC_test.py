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

from SVC_model import PCA_Find_ncomp, PCA_transform, SVM_Gridsearch, SVC_Linear, SVC_rbf
from functions import *
from sklearn.utils import class_weight
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle

SSD_path = "/home/ens/AN03460/Desktop/TP4/music/music/tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"
#
X, Y, id,le = get_data(SSD_path)
X = preprocessing.normalize(X, norm='max', axis=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=60, stratify=Y)
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_train),
                                                  Y_train)
#
#
N_comp = PCA_Find_ncomp(X_train, 0.95)
#
# print(N_comp)
#
PCA_X_Train, PCA_X_Test = PCA_transform(X_train, X_test, N_comp)
#
#
#
# # Grid = SVM_Gridsearch(PCA_X_Train[:1000], Y_train[:1000])
#
# print('best param')
# print(Grid.best_params_)
#
#
# print('best score')
# print(Grid.best_score_)
# print(Grid.cv_results_)
# print(Grid)
#
# result = Grid.cv_results_
# df = pd.DataFrame(data=result)
# dfData = df[['param_kernel', 'param_C', 'param_gamma', 'rank_test_Accuracy', 'mean_test_Accuracy', 'std_test_Accuracy',
#              'mean_test_F1', 'std_test_F1', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']]
# dfData = dfData.sort_values(['param_kernel', 'rank_test_Accuracy', 'mean_test_F1'], ascending=[True, True, True])
# dfData
#SVC_Linear(PCA_X_Train, Y_train, PCA_X_Test, Y_test, 1)

# Svc_ssd,y_pred,train_time,pred_time=SVC_rbf(PCA_X_Train[:1000], Y_train[:1000], PCA_X_Test[:1000], Y_test[:1000], 10, 1)
# print(Svc_ssd)
# pickle.dump(Svc_ssd,open('svm_ssd.sav','wb'))
# loaded_model =pickle.load(open('svm_ssd.sav','rb'))
#
# svm_result_ssd=loaded_model.score(PCA_X_Test[:1000],Y_test[:1000])
#
# print(loaded_model.predict(PCA_X_Test[:1000]))
# print('pickle result')
# print (svm_result_ssd)

# pickle.dump(Svc_marsyas,open('svm_marsyas.sav','wb'))
# loaded_model =pickle.load(open('svm_marsyas.sav','rb'))
#
# svm_result_marsyas=loaded_model.score(PCA_X_Test,Y_test)
#
# print(loaded_model.predict(PCA_X_Test))
# print (svm_result_marsyas)

# dataset_path=("/home/ens/AN03460/Desktop/TP4/music/music/tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv")
# print('Test ACC marsvas')
# X, Y,le = get_data(dataset_path)
# X = preprocessing.normalize(X, norm='max',axis = 0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=60, stratify=Y)
# N_comp=PCA_Find_ncomp(X_train,0.999)
# print(N_comp)


# dataset_path=("/home/ens/AN03460/Desktop/TP4/music/music/tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv")
# print('Test ACC mfccs')
# X, Y,le = get_data(dataset_path)
# X = preprocessing.normalize(X, norm='max',axis = 0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=60, stratify=Y)
# N_comp=PCA_Find_ncomp(X_train,0.95)
# print(N_comp)




#PCA_X_Train,PCA_X_Test=PCA_transform(X_train,X_test,N_comp)

#Grid = SVM_Gridsearch(X_train[:30000], Y_train[:30000])

#print('best param')
#print(Grid.best_params_)


#print('best score')
#print(Grid.best_score_)

Svc_ssd,y_pred,train_time,pred_time=SVC_rbf(PCA_X_Train[:2000], Y_train[:2000], PCA_X_Test[:2000], Y_test[:2000], 10, 1)
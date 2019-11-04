#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Course :
GTI770 — Systèmes intelligents et apprentissage machine

Project :
Lab # 3 — Machines à vecteur de support et réseaux neuronaux

Students :
Alexendre Bleau — BLEA14058906
David Létinaud  — LETD05129708
Thomas Lioret   — LIOT20069605

Group :
GTI770-A19-01
"""

from RN_model import *
from functions import get_data , plot_perf
import time
import matplotlib.pyplot as plt
#from sklearn.metrics import f1_score, accuracy_score, recall_score, average_precision_score

X_train, X_test, Y_train, Y_test = get_data()

layer_sizes = [100, 100, 2]
epochs = 60
learning_rate = 0.0005
batch_size = 100

dropout = 0.5




# model = RN_model(layer_sizes, dropout, learning_rate)
# model.fit(X_train, Y_train, batch_size = 100, epochs = 200)
# print('f1 score: {}'.format(f1_score(Y_test, np.where(model.predict(X_test) > 0.5, 1, 0))))

## Questions à poser :  ligne 45 de RN_model : PK input_dim=77 , n'a aucun effet
##                      val_loss/val_acc
# A FAIRE : mesure du temps et accuracy et F1score et le plot


################################## Nombres de couche

################################## Nombres de perceptrons

################################## Nombres d'iterations

################################## Learning rate 
training_delay_RN = []
predicting_delay_RN = []
perf_RN = []
best_index_RN = 0
best_y_test_RN =  []
history_obj = []
#l_rate_range = np.arange(0.0001,0.04,0.0005) #A garder
# l_rate_range = np.logspace(0.0001, 0.004, 1, endpoint=False)
#l_rate_range = [0.000001,0.00005, 0.0005, 0.001, 0.01, 0.02, 0.03, 0.05]
#l_rate_range = [0.00001,0.0005,0.001]
l_rate_range = [0.0005]

cpt = 0
best_accuracy_RN = 0
for l_rate in l_rate_range:
    model = RN_model(layer_sizes, dropout, l_rate)
    #### Apprentissage
    start = time.time()
    #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
    hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test))
    
    end = time.time()
    training_delay_RN.append(end - start)

    history_obj.append( list(hist_obj.history.values()))

    #### Prédiction
    start = time.time()
    
    Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)

    end = time.time()
    predicting_delay_RN.append(end - start)

    # Calcul Perfs
    # Y_pred = np.argmax(Y_pred, axis = 1)  # Reshape probas vector TO number of the max proba
    # perf = perf_mesure(Y_pred, Y_test)
    # perf_RN.append(perf)

    # if perf[0]> best_accuracy_RN:
    #     best_accuracy_RN = perf[0]
    #     best_index_RN = cpt
    #     best_y_pred_RN =  Y_pred
    # cpt+=1
    # print("l_rate : ",l_rate, "perf : ", perf)

# Best Perf :
# print("Best accuracy : {} for learning_rate = {}".format(perf_RN[best_index_RN][0] , l_rate_range[best_index_RN] ) )
# print("Learning delay : {} | predicting delay = {}".format(training_delay_RN[best_index_RN] , predicting_delay_RN[best_index_RN] ) )

ho = np.array(history_obj)
ho = ho.transpose(1,2,0) 
leg = [str(i) for i in l_rate_range]
title_ = ['loss','acc','val_loss','val_acc']
x_lab = "epochs"


plot_perf(ho, title_, "RN : étude du learning_rate ",title_)



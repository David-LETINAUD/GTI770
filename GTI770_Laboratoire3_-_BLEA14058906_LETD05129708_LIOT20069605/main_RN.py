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
from functions import get_data , plot_perf, plot_delay
import time
import matplotlib.pyplot as plt
#from sklearn.metrics import f1_score, accuracy_score, recall_score, average_precision_score

X_train, X_test, Y_train, Y_test = get_data()

layer_sizes = [100, 100, 2]
epochs = 60
learning_rate = 0.0005
batch_size = 100

dropout = 0.5


training_delay_RN = []
predicting_delay_RN = []
history_obj = []
# Faire 1 test à la fois ou réinitialiser les 3 lists

################################## Nombres de couches

################################## Nombres de perceptrons

################################## Nombres d'iterations

################################## Learning rate 

#l_rate_range = np.arange(0.0001,0.04,0.0005) #A garder
# l_rate_range = np.logspace(0.0001, 0.004, 1, endpoint=False)
#l_rate_range = [0.000001,0.00005, 0.0005, 0.001, 0.01, 0.02, 0.03, 0.05]
l_rate_range = [0.00001,0.0005,0.001]
#l_rate_range = [0.0005]

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

ho = np.array(history_obj)
ho = ho.transpose(1,2,0) 
leg = [str(i) for i in l_rate_range]
sub_title = ['loss','acc','f1','val_loss','val_acc', 'val_f1']
x_lab = "epochs"


plot_perf(ho, leg, "RN : étude du learning_rate ",sub_title)

plot_delay(training_delay_RN,predicting_delay_RN,"RN : HyperParam = learning_rate")

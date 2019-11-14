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


from tensorflow.keras.callbacks import TensorBoard
import shutil

X_train, X_test, Y_train, Y_test = get_data()

layer_sizes = [100, 100, 2]
epochs = 60
learning_rate = 0.0005
batch_size = 100

dropout = 0.5


training_delay_RN = []
predicting_delay_RN = []
history_obj = []

best_accuracy_RN = 0
# Faire 1 test à la fois ou réinitialiser les 3 lists

################################## Nombres de couches cachees
layer_sizes_range = [[100],[100, 100, 2],[100, 100, 100, 100, 100, 2]]

try:
    shutil.rmtree('./logs')
except:
    print("nothing to delete")
    
tensorboard_callback = []
for i in range(3):
    tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))#time.time())))
# Par invité de commande : 
# tensorboard --logdir="./logs" --port 6006
cpt = 0
for layer_s in layer_sizes_range:
    model = RN_model(layer_s, dropout, learning_rate)
    #### Apprentissage
    start = time.time()
    #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
    hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])
    
    end = time.time()
    training_delay_RN.append(end - start)

    history_obj.append( list(hist_obj.history.values()))

    #### Prédiction
    start = time.time()
    
    Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)

    end = time.time()
    predicting_delay_RN.append(end - start)
    cpt+=1
################################## Nombres de perceptrons
# layer_sizes_range = [[5, 4, 4],[100, 100, 2],[500, 500, 500]]
#
# try:
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
    
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))#time.time())))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for layer_s in layer_sizes_range:
#     model = RN_model(layer_s, dropout, learning_rate)
#     #### Apprentissage
#     start = time.time()
#     #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])
#
#     end = time.time()
#     training_delay_RN.append(end - start)
#
#     history_obj.append( list(hist_obj.history.values()))
#
#     #### Prédiction
#     start = time.time()
#
#     Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
#
#     end = time.time()
#     predicting_delay_RN.append(end - start)

################################## Nombres d'iterations
# epochs_range = [30,60, 120]#[10,60,500]
# max_ep = max(epochs_range)
#
# try:
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
    
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))#time.time())))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for ep in epochs_range:
#     model = RN_model(layer_sizes, dropout, learning_rate)
#     #### Apprentissage
#     start = time.time()
#     #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = ep, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])
#
#     end = time.time()
#     training_delay_RN.append(end - start)
#
#     ho_tmp = list(hist_obj.history.values())
#     ho_tmp = [i + [np.nan for _ in range(max_ep-ep)] for i in ho_tmp ]
#     history_obj.append(ho_tmp)
#
#
#     #### Prédiction
#     start = time.time()
#
#     Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
#
#     end = time.time()
#     predicting_delay_RN.append(end - start)

################################## Learning rate 
#l_rate_range = [0.00001,0.0005,0.1]
#l_rate_range = [0.0005]

# try:
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
    
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))#time.time())))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for l_rate in l_rate_range:
#     model = RN_model(layer_sizes, dropout, l_rate)
#     #### Apprentissage
#     start = time.time()
#     #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])
#
#     end = time.time()
#     training_delay_RN.append(end - start)
#
#     history_obj.append( list(hist_obj.history.values()))
#
#     #### Prédiction
#     start = time.time()
#
#     Y_pred = np.where(model.predict(X_test) > 0.5, 1, 0)
#
#     end = time.time()
#     predicting_delay_RN.append(end - start)
#     cp+=1


# A faire sauf pour epochs
ho = np.array(history_obj)
ho = ho.transpose(1,2,0) 


sub_title = ['loss','acc','f1','val_loss','val_acc', 'val_f1']
x_lab = "epochs"


leg = [str(i) for i in layer_sizes_range]
#leg = [str(i) for i in epochs_range]
#leg = [str(i) for i in l_rate_range]

titre = "RN : HyperParam = number of layer"
#titre = "RN : HyperParam = layer size"
#titre = "RN : HyperParam = number of epochs"
#titre = "RN : HyperParam = learning rate"

plot_perf(ho, leg, titre ,sub_title)
plot_delay(training_delay_RN,predicting_delay_RN,titre)

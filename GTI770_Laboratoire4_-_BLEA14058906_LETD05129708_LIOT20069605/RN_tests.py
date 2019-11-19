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

from functions import *
from RN_model import RN_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import shutil
import time

MFCC_path = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv"

X, Y = get_data(MFCC_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

nb_features = len(X[0])
nb_classes = 25 #max(Y)+1

train_size = 1000

layer_sizes = [100, 100, 2]
epochs = 60
learning_rate = 0.0005
batch_size = 100

dropout = 0.5

# Pour affichage
sub_title = ['loss','acc','f1','val_loss','val_acc', 'val_f1']
x_lab = "epochs"

################################## Nombres de couches cachees
training_delay_RN = []
predicting_delay_RN = []
history_obj = []
cpt = 0
best_accuracy_RN = 0

layer_sizes_range = [[50]]#,[100, 2],[100, 100, 2]]

# Suppression de la dernière étude d'hyperparamètre

try:      
    shutil.rmtree('./logs')
except:
    print("nothing to delete")
# Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
tensorboard_callback = []
for i in range(3):
    tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# Par invité de commande : 
# tensorboard --logdir="./logs" --port 6006
cpt = 0
for layer_s in layer_sizes_range:
    model = RN_model(layer_s, dropout, learning_rate, nb_features, nb_classes)
    #### Apprentissage                                                                                                                                                               
    start = time.time()                                                                                                                   
    hist_obj = model.fit(X_train[0:train_size], Y_train[0:train_size], batch_size = batch_size, epochs = 3, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]]) 

    end = time.time()
    training_delay_RN.append(end - start)

    history_obj.append( list(hist_obj.history.values()))

    #### Prédiction                                                                                                                                                                  
    start = time.time()

    Y_pred_temp = model.predict(X_test)
    Y_pred = []
    for i in Y_pred_temp:
        tmp = np.zeros(nb_classes)
        tmp[np.argmax(i)] = 1
        Y_pred.append(tmp)
    #print(Y_pred)

    end = time.time()
    predicting_delay_RN.append(end - start)
    cpt+=1

# Mise en forme des données pour l'affichage
ho = np.array(history_obj)
ho = ho.transpose(1,2,0)

leg = [str(i) for i in layer_sizes_range]                                                                                                                                              

titre = "RN : HyperParam = number of layer"                                                                                                                                         

plot_perf(ho, leg, titre ,sub_title)
plot_delay(training_delay_RN,predicting_delay_RN,titre)

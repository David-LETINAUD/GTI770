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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import shutil
import time

from sklearn import metrics
from sklearn.utils import class_weight
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# The others
dataset_path_tab = []
dataset_path_tab.append("./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv") # best =>31%
dataset_path_tab.append("./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv")
dataset_path_tab.append("./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv") # 2nd => 27%

layer_sizes = [500]
epochs = 100
learning_rate = 0.001
batch_size = 500

dropout = 0.5

sub_title = ['loss','acc','val_loss','val_acc']


################################## Dataset TEST
training_delay_RN = []
predicting_delay_RN = []
history_obj = []
cpt = 0
best_accuracy_RN = 0
f1_RN = []
acc_RN = []

#layer_sizes_range = [[500]] #,[100, 20],[100, 100, 20]]

try:      
    shutil.rmtree('./logs')
except:
    print("nothing to delete")
# Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
tensorboard_callback = []
for i in range(len(dataset_path_tab)):
    tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# Par invité de commande : 
# tensorboard --logdir="./logs" --port 6006

# Save keras model
checkpoint_path = []
checkpoint_path.append("Models/MLP_model_SSD/cp.ckpt")
checkpoint_path.append("Models/MLP_model_MFCC/cp.ckpt")
checkpoint_path.append("Models/MLP_model_MARSYAS/cp.ckpt")

checkpoint_dir = []
cp_callback = []
for cp in checkpoint_path:
    checkpoint_dir.append(os.path.dirname(cp))
    # Create a callback that saves the model's weights
    cp_callback.append(tf.keras.callbacks.ModelCheckpoint(filepath=cp,
                                                    save_weights_only=True,
                                                    verbose=1))

cpt = 0
for path_ in dataset_path_tab:
    X, Y, le = get_data(path_)
    class_names = list(le.classes_)
    X = preprocessing.normalize(X, norm='max',axis = 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

    class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(Y_train),
                                                 Y_train)
    print(class_weights)

    nb_features = len(X[0])
    nb_classes = max(Y)+1
    train_size = len(X)

    model = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
    #### Apprentissage                                                                                                                                                               
    start = time.time()                                                                                                                   
    hist_obj = model.fit(X_train[0:train_size], Y_train[0:train_size], batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt],cp_callback[cpt]], class_weight=class_weights) 
    end = time.time()

    training_delay_RN.append(end - start)
    history_obj.append( list(hist_obj.history.values()))

    #### Prédiction                                                                                                                                                                  
    start = time.time()
    Y_pred_temp = model.predict(X_test)
    end = time.time()
    predicting_delay_RN.append(end - start)

    # remise en forme de Y_pred
    Y_pred = []
    for i in Y_pred_temp:
        Y_pred.append(np.argmax(i))    
    
    f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
    acc = metrics.accuracy_score(Y_test, Y_pred)
    print("acc :", acc,"f1 :", f1)
    #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
    f1_RN.append(f1)
    acc_RN.append(acc)
    cpt+=1

#### Prédiction            
# Loads the weights
print("NEW MODEL")
model2 = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
print(checkpoint_path)
model2.load_weights(checkpoint_path[2])

# Re-evaluate the model
#loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
Y_pred_temp2 = model2.predict(X_test)

# remise en forme de Y_pred
Y_pred2 = []
for i in Y_pred_temp2:
    Y_pred2.append(np.argmax(i))    

f1 = metrics.f1_score(Y_test, Y_pred2,average='weighted')
acc = metrics.accuracy_score(Y_test, Y_pred2)
print("acc :", acc,"f1 :", f1)
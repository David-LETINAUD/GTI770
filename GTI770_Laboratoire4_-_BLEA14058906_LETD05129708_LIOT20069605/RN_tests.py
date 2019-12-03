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

# TOP 3 selected
#data_path = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv" # => MLP 22%
data_path = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv" #=> MLP 30.7%
#data_path = "./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv" #=> MLP 20%
#data_path = "./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv"

# The others
dataset_path_tab = []
dataset_path_tab.append("./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv") # best =>31%
#dataset_path_tab.append("./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv")
#dataset_path_tab.append("./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv")
#dataset_path_tab.append("./tagged_feature_sets/msd-jmirderivatives_dev/msd-jmirderivatives_dev.csv") # 3rd => 25%
# dataset_path_tab.append("./tagged_feature_sets/msd-jmirlpc_dev/msd-jmirlpc_dev.csv")
# dataset_path_tab.append("./tagged_feature_sets/msd-jmirmoments_dev/msd-jmirmoments_dev.csv")
#dataset_path_tab.append("./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv") # 2nd => 27%
# dataset_path_tab.append("./tagged_feature_sets/msd-mvd_dev/msd-mvd_dev.csv")
# dataset_path_tab.append("./tagged_feature_sets/msd-rh_dev_new/msd-rh_dev_new.csv")
# dataset_path_tab.append("./tagged_feature_sets/msd-trh_dev/msd-trh_dev.csv")



X, Y,id, le = get_data(data_path)
X = preprocessing.normalize(X, norm='max',axis = 0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

# print(list(le.inverse_transform(Y[:10])))
class_names = list(le.classes_)

nb_features = len(X[0])
nb_classes = max(Y)+1
train_size = 1000#len(X)

X_train = X_train[:train_size]
Y_train = Y_train[:train_size]

layer_sizes = [500]
epochs = 5 #50
learning_rate = 0.0005
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
checkpoint_path = "MLP_model/MLP_model_SSD/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# cpt = 0
# for path_ in dataset_path_tab:
#     X, Y, le = get_data(path_)
#     X = preprocessing.normalize(X, norm='max',axis = 0)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

#     class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(Y_train),
#                                                  Y_train)
#     print(class_weights)

#     nb_features = len(X[0])
#     nb_classes = max(Y)
#     train_size = len(X)

#     model = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
#     #### Apprentissage                                                                                                                                                               
#     start = time.time()                                                                                                                   
#     hist_obj = model.fit(X_train[0:train_size], Y_train[0:train_size], batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt],cp_callback], class_weight=class_weights, callbacks = [tensorboard_callback[cpt],cp_callback]) 
#     end = time.time()

#     training_delay_RN.append(end - start)
#     history_obj.append( list(hist_obj.history.values()))

#     #### Prédiction                                                                                                                                                                  
#     start = time.time()
#     Y_pred_temp = model.predict(X_test)
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # remise en forme de Y_pred
#     Y_pred = []
#     for i in Y_pred_temp:
#         # tmp = np.zeros(nb_classes)
#         # tmp[np.argmax(i)] = 1
#         #Y_pred.append(tmp)

#         Y_pred.append(np.argmax(i))    
    

#     #print(metrics.confusion_matrix(Y_test, Y_pred))
#     #print(metrics.classification_report(Y_test, Y_pred, digits=3)) 
#     f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
#     acc = metrics.accuracy_score(Y_test, Y_pred)
#     print("acc :", acc,"f1 :", f1)
#     #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
#     f1_RN.append(f1)
#     acc_RN.append(acc)
#     cpt+=1

# # Mise en forme des données pour l'affichage
# ho = np.array(history_obj)
# ho = ho.transpose(1,2,0)                                                                                                            

# # Pour affichage
# sub_title = ['loss','acc','val_loss','val_acc']
# x_lab = "epochs"
# leg = [str(i) for i in range(len(dataset_path_tab))]  
# titre = "RN : Dataset test"                                                                                                                                         

# plot_perf_epochs(ho, leg, titre ,sub_title)
# plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)
# plot_confusion_matrix(Y_test,Y_pred,class_names, title="TEST {}".format(0))

#### Prédiction            
# Loads the weights
print("NEW MODEL")
model2 = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
model2.load_weights(checkpoint_path)

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



################################## Nombres de couches cachees
# training_delay_RN = []
# predicting_delay_RN = []
# history_obj = []
# cpt = 0
# best_accuracy_RN = 0
# f1_RN = []
# acc_RN = []


# layer_sizes_range = [[500],[500, 250],[500, 250, 125]]

# # Suppression de la dernière étude d'hyperparamètre
# try:      
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
# # Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for layer_s in layer_sizes_range:
#     model = RN_model(layer_s, dropout, learning_rate, nb_features, nb_classes)
#     #### Apprentissage                                                                                                                                                               
#     start = time.time()                                                                                                                   
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]]) 

#     end = time.time()
#     training_delay_RN.append(end - start)

#     history_obj.append( list(hist_obj.history.values()))

#     #### Prédiction                                                                                                                                                                  
#     start = time.time()
#     Y_pred_temp = model.predict(X_test)
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # remise en forme de Y_pred
#     Y_pred = []
#     for i in Y_pred_temp:
#         Y_pred.append(np.argmax(i))  

#     f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
#     acc = metrics.accuracy_score(Y_test, Y_pred)
#     print("acc :", acc,"f1 :", f1)
#     #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
#     f1_RN.append(f1)
#     acc_RN.append(acc)

#     cpt+=1

# # Mise en forme des données pour l'affichage
# ho = np.array(history_obj)
# ho = ho.transpose(1,2,0)

# leg = [str(i) for i in layer_sizes_range]                                                                                                                                              

# titre = "RN : HyperParam = number of layer"                                                                                                                                         

# plot_perf_epochs(ho, leg, titre ,sub_title)
# plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)

# plot_confusion_matrix(Y_test,Y_pred,class_names, title="TEST {}".format(0))

# ################################## Nombres de perceptrons
# training_delay_RN = []
# predicting_delay_RN = []
# history_obj = []

# best_accuracy_RN = 0

# nb_perceptrons_range = [[50],[100],[500]]                                                                                                                      

# # Suppression de la dernière étude d'hyperparamètre
# try:    
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
# # Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for nb_perceptrons in nb_perceptrons_range:                                                                                                                                                  
#     model = RN_model(nb_perceptrons, dropout, learning_rate, nb_features, nb_classes)                                                                                                                             
#     #### Apprentissage                                                                                                                                                             
#     start = time.time()                                                                                                                                                            
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])                                                             

#     end = time.time()                                                                                                                                                              
#     training_delay_RN.append(end - start)                                                                                                                                          

#     history_obj.append( list(hist_obj.history.values()))

#     #### Prédiction                                                                                                                                                                  
#     start = time.time()
#     Y_pred_temp = model.predict(X_test)
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # remise en forme de Y_pred
#     Y_pred = []
#     for i in Y_pred_temp:
#         Y_pred.append(np.argmax(i))  

#     f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
#     acc = metrics.accuracy_score(Y_test, Y_pred)
#     print("acc :", acc,"f1 :", f1)
#     #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
#     f1_RN.append(f1)
#     acc_RN.append(acc)
#     cpt+=1   

# # Mise en forme des données pour l'affichage final
# ho = np.array(history_obj)
# ho = ho.transpose(1,2,0)

# leg = [str(i) for i in nb_perceptrons_range]  

# titre = "RN : HyperParam = layer size"                                                                                                                                          

# plot_perf_epochs(ho, leg, titre ,sub_title)
# plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)

# ################################## Nombres d'iterations
# training_delay_RN = []
# predicting_delay_RN = []
# history_obj = []
# cpt = 0
# best_accuracy_RN = 0

# epochs_range = [30,60, 5000]                                                                                                                                            
# max_ep = max(epochs_range) 

# # Suppression de la dernière étude d'hyperparamètre
# try:
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")
# # Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for ep in epochs_range:                                                                                                                                                            
#     model = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)                                                                                                                       
#     #### Apprentissage                                                                                                                                                             
#     start = time.time()                                                                                                                                                            
#     #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)                                                                                                                    
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = ep, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])                                                                  

#     end = time.time()                                                                                                                                                              
#     training_delay_RN.append(end - start)                                                                                                                                          

#     ho_tmp = list(hist_obj.history.values())                                                                                                                                       
#     ho_tmp = [i + [np.nan for _ in range(max_ep-ep)] for i in ho_tmp ]                                                                                                             
#     history_obj.append(ho_tmp)

#     #### Prédiction                                                                                                                                                                  
#     start = time.time()
#     Y_pred_temp = model.predict(X_test)
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # remise en forme de Y_pred
#     Y_pred = []
#     for i in Y_pred_temp:
#         Y_pred.append(np.argmax(i))  

#     f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
#     acc = metrics.accuracy_score(Y_test, Y_pred)
#     print("acc :", acc,"f1 :", f1)
#     #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
#     f1_RN.append(f1)
#     acc_RN.append(acc)
#     cpt+=1

# # Mise en forme des données pour l'affichage final
# ho = np.array(history_obj)
# ho = ho.transpose(1,2,0)

# leg = [str(i) for i in epochs_range]                                                                                                                                                
                                                                                                                                       
# titre = "RN : HyperParam = number of epochs"                                                                                                                                          

# plot_perf_epochs(ho, leg, titre ,sub_title)
# plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)

# ################################## Learning rate 
# training_delay_RN = []
# predicting_delay_RN = []
# history_obj = []
# cpt = 0
# best_accuracy_RN = 0

# l_rate_range = [0.00001,0.0005,0.01]

# # Suppression de la dernière étude d'hyperparamètre
# try:
#     shutil.rmtree('./logs')
# except:
#     print("nothing to delete")

# # Callbacks pour affichage des performances dans tensorflow : 1 callback pour chaque hyperparamètre
# tensorboard_callback = []
# for i in range(3):
#     tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# # Par invité de commande : 
# # tensorboard --logdir="./logs" --port 6006
# cpt = 0
# for l_rate in l_rate_range:
#     model = RN_model(layer_sizes, dropout, l_rate, nb_features, nb_classes)
#     #### Apprentissage
#     start = time.time()
#     #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
#     hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]])
#     end = time.time()
#     training_delay_RN.append(end - start)

#     history_obj.append( list(hist_obj.history.values()))

#     #### Prédiction                                                                                                                                                                  
#     start = time.time()
#     Y_pred_temp = model.predict(X_test)
#     end = time.time()
#     predicting_delay_RN.append(end - start)

#     # remise en forme de Y_pred
#     Y_pred = []
#     for i in Y_pred_temp:
#         Y_pred.append(np.argmax(i))  

#     f1 = metrics.f1_score(Y_test, Y_pred,average='weighted')
#     acc = metrics.accuracy_score(Y_test, Y_pred)
#     print("acc :", acc,"f1 :", f1)
#     #  Assurez-vous d’avoir le paramètre average=‘weighted’ afin de pondérer correctement le score en fonction du nombre d’instances de chaque classe.
#     f1_RN.append(f1)
#     acc_RN.append(acc)
#     cpt+=1
# # Traitement pour affichage
# ho = np.array(history_obj)
# ho = ho.transpose(1,2,0)
                                                                                                                                      
# leg = [str(i) for i in l_rate_range]                                                                                                                                                
                                                                                                                                    
# titre = "RN : HyperParam = learning rate"                                                                                                                                           

# plot_perf_epochs(ho, leg, titre ,sub_title)
# plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)

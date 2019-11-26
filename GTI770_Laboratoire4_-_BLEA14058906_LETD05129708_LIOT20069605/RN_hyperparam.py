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
#from imblearn.under_sampling import RandomUnderSampler


data_path = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv"

X, Y, le = get_data(data_path)
X = preprocessing.normalize(X, norm='max',axis = 0)

# rus = RandomUnderSampler(random_state=0)
# X, Y = rus.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

class_weights = class_weight.compute_class_weight('balanced',
                                                np.unique(Y_train),
                                                Y_train)
print(class_weights)

# print(list(le.inverse_transform(Y[:10])))
class_names = list(le.classes_)

nb_features = len(X[0])
nb_classes = max(Y)+1
train_size = len(X)


X_train = X_train[:train_size]
Y_train = Y_train[:train_size]

# Parametres de base
layer_sizes = [500] # OK
epochs = 100 # OK avec 100
learning_rate = 0.001
batch_size = 500

dropout = 0.5

sub_title = ['loss','acc','val_loss','val_acc']


ho = 0
leg = []
titre = ""
f1_RN = []
acc_RN = []
training_delay_RN = []
predicting_delay_RN = []


########################## Layer size hyperparam test
def layer_size_test():
    global training_delay_RN
    training_delay_RN = []
    global predicting_delay_RN 
    predicting_delay_RN = []
    history_obj = []
    cpt = 0
    global f1_RN
    f1_RN = []
    global acc_RN
    acc_RN = []
    
    global ho
    global leg
    global titre

    layer_sizes_range = [[500],[500, 250],[500, 250, 125]]

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
        hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]],class_weight=class_weights) 

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
    # Mise en forme des données pour l'affichage
    ho = np.array(history_obj)
    ho = ho.transpose(1,2,0)

    leg = [str(i) for i in layer_sizes_range]                                                                                                                                              

    titre = "RN : HyperParam = number of layer"       

################################## Nombres de perceptrons
def perceptrons_number_test():
    global training_delay_RN
    training_delay_RN = []
    global predicting_delay_RN 
    predicting_delay_RN = []
    history_obj = []
    cpt = 0
    global f1_RN
    f1_RN = []
    global acc_RN
    acc_RN = []
    
    global ho
    global leg
    global titre

    nb_perceptrons_range = [[100],[500],[1000]]                                                                                                                      

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
    for nb_perceptrons in nb_perceptrons_range:                                                                                                                                                  
        model = RN_model(nb_perceptrons, dropout, learning_rate, nb_features, nb_classes)                                                                                                                             
        #### Apprentissage                                                                                                                                                             
        start = time.time()                                                                                                                                                            
        hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]],class_weight=class_weights)                                                             

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
    # Mise en forme des données pour l'affichage final
    ho = np.array(history_obj)
    ho = ho.transpose(1,2,0)

    leg = [str(i) for i in nb_perceptrons_range]  

    titre = "RN : HyperParam = layer size"    

################################## Learning rate 
def learning_rate_test():
    global training_delay_RN
    training_delay_RN = []
    global predicting_delay_RN 
    predicting_delay_RN = []
    history_obj = []
    cpt = 0
    global f1_RN
    f1_RN = []
    global acc_RN
    acc_RN = []
    
    global ho
    global leg
    global titre

    l_rate_range = [0.0005, 0.001,0.005]
    print(l_rate_range)

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
    for l_rate in l_rate_range:
        model = RN_model(layer_sizes, dropout, l_rate, nb_features, nb_classes)
        #### Apprentissage
        start = time.time()
        #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)
        hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]],class_weight=class_weights)
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
        
    # Traitement pour affichage
    ho = np.array(history_obj)
    ho = ho.transpose(1,2,0)
                                                                                                                                        
    leg = [str(i) for i in l_rate_range]                                                                                                                                                
                                                                                                                                        
    titre = "RN : HyperParam = learning rate" 

################################## Nombres d'iterations
def epochs_number_test():
    global training_delay_RN
    training_delay_RN = []
    global predicting_delay_RN 
    predicting_delay_RN = []
    history_obj = []
    cpt = 0
    global f1_RN
    f1_RN = []
    global acc_RN
    acc_RN = []

    global ho
    global leg
    global titre

    epochs_range = [10,100, 1000]                                                                                                                                            
    max_ep = max(epochs_range) 

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

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(Y_train),
                                                      Y_train)
    print(class_weights)


    cpt = 0
    for ep in epochs_range:                                                                                                                                                            
        model = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)                                                                                                                       
        #### Apprentissage                                                                                                                                                             
        start = time.time()                                                                                                                                                            
        #model.fit(X_train, Y_train, batch_size = 100, epochs = 60)                                                                                                                    
        hist_obj = model.fit(X_train, Y_train, batch_size = batch_size, epochs = ep, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]],class_weight=class_weights)

        end = time.time()                                                                                                                                                              
        training_delay_RN.append(end - start)                                                                                                                                          

        ho_tmp = list(hist_obj.history.values())                                                                                                                                       
        ho_tmp = [i + [np.nan for _ in range(max_ep-ep)] for i in ho_tmp ]                                                                                                             
        history_obj.append(ho_tmp)

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
    # Mise en forme des données pour l'affichage final
    ho = np.array(history_obj)
    ho = ho.transpose(1,2,0)

    leg = [str(i) for i in epochs_range]                                                                                                                                                
                                                                                                                                        
    titre = "RN : HyperParam = number of epochs" 

def RN_plot_test():
    #print( leg, titre ,sub_title)
    plot_perf_epochs(ho, leg, titre ,sub_title)
    plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)
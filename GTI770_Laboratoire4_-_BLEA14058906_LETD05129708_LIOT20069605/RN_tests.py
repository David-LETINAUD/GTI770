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
from sklearn.model_selection import train_test_split
import shutil
import time

from sklearn import metrics

# TOP 3 selected
#MFCC_path = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv" # => MLP 22%
MFCC_path = "./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv" #=> MLP 30.7%
#MFCC_path = "./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv" #=> MLP 20%

# The others
direct_path_tab = []
direct_path_tab.append("./tagged_feature_sets/msd-ssd_dev/msd-ssd_dev.csv") # best =>31%
direct_path_tab.append("./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv")
direct_path_tab.append("./tagged_feature_sets/msd-jmirspectral_dev/msd-jmirspectral_dev.csv")
# direct_path_tab.append("./tagged_feature_sets/msd-jmirderivatives_dev/msd-jmirderivatives_dev.csv") # 3rd => 25%
# direct_path_tab.append("./tagged_feature_sets/msd-jmirlpc_dev/msd-jmirlpc_dev.csv")
# direct_path_tab.append("./tagged_feature_sets/msd-jmirmoments_dev/msd-jmirmoments_dev.csv")
# direct_path_tab.append("./tagged_feature_sets/msd-marsyas_dev_new/msd-marsyas_dev_new.csv") # 2nd => 27%
# direct_path_tab.append("./tagged_feature_sets/msd-mvd_dev/msd-mvd_dev.csv")
# direct_path_tab.append("./tagged_feature_sets/msd-rh_dev_new/msd-rh_dev_new.csv")
# direct_path_tab.append("./tagged_feature_sets/msd-trh_dev/msd-trh_dev.csv")

leg = [str(i) for i in range(len(direct_path_tab))]  

# X, Y = get_data(MFCC_path)
# X = preprocessing.normalize(X, norm='max',axis = 0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

# nb_features = len(X[0])
# nb_classes = max(Y)+1

# train_size = len(X)

layer_sizes = [500]
epochs = 50
learning_rate = 0.0005
batch_size = 1000

dropout = 0.5

# Pour affichage
sub_title = ['loss','acc','val_loss','val_acc']
x_lab = "epochs"

################################## Nombres de couches cachees
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
for i in range(len(direct_path_tab)):
    tensorboard_callback.append(TensorBoard(log_dir="logs\{}".format(i)))
# Par invité de commande : 
# tensorboard --logdir="./logs" --port 6006
cpt = 0
for path_ in direct_path_tab:
    X, Y = get_data(path_)
    X = preprocessing.normalize(X, norm='max',axis = 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,random_state=60, stratify=Y)  # 70% training and 30% test

    nb_features = len(X[0])
    nb_classes = max(Y)+1
    train_size = len(X)

    model = RN_model(layer_sizes, dropout, learning_rate, nb_features, nb_classes)
    #### Apprentissage                                                                                                                                                               
    start = time.time()                                                                                                                   
    hist_obj = model.fit(X_train[0:train_size], Y_train[0:train_size], batch_size = batch_size, epochs = epochs, validation_data=(X_test, Y_test), callbacks = [tensorboard_callback[cpt]]) 
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
        # tmp = np.zeros(nb_classes)
        # tmp[np.argmax(i)] = 1
        #Y_pred.append(tmp)

        Y_pred.append(np.argmax(i))    
    

    #print(metrics.confusion_matrix(Y_test, Y_pred))
    #print(metrics.classification_report(Y_test, Y_pred, digits=3)) 
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

titre = "RN : Dataset test"                                                                                                                                         

plot_perf_epochs(ho, leg, titre ,sub_title)
plot_perf_delay(f1_RN,acc_RN,training_delay_RN,predicting_delay_RN,titre)
#plot_delay(training_delay_RN,predicting_delay_RN,titre)

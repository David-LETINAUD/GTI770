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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
import numpy as np


# Inspiré de : https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    """
    Calcul du rappel     
    input :
        y_true (tensorflow.python.framework.ops.Tensor') : sorties réelles
        y_pred (tensorflow.python.framework.ops.Tensor') : sorties prédites
    output : 
        (tensorflow.python.framework.ops.Tensor') : rappel    
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Calcul de la précision     
    input :
        y_true (tensorflow.python.framework.ops.Tensor')  : sorties réelles
        y_pred (tensorflow.python.framework.ops.Tensor')  : sorties prédites
    output : 
        (tensorflow.python.framework.ops.Tensor')  : précision    
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    """
    Calcul du f1-score     
    input :
        y_true (tensorflow.python.framework.ops.Tensor')  : sorties réelles
        y_pred (tensorflow.python.framework.ops.Tensor')  : sorties prédites
    output : 
        (tensorflow.python.framework.ops.Tensor')  : f1-score    
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Inspiré de : https://github.com/LeighWeston86/multilayer_perceptron
def RN_model(layer_sizes, dropout, learning_rate):
    """
    Multilayer perceptron for binary classification.
    
    input :
        layer_sizes (list) : size for each hidden layer
        dropout (float) : dropout for hidden layers
        learning_rate (float) : learning rate for Adam optmizer
    output : 
        (keras model) : compiled model for multilayer perceptron
    
    """
    with tf.device('/GPU:0'):

        model = Sequential()

        # Couche d'entrée
        model.add(Dense(77))  
        model.add(Activation('relu'))

        # Couches cachées
        for size in layer_sizes:
            model.add(Dense(size))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))

        # Couche de sortie
        model.add(Dense(1))
        model.add(Activation('sigmoid'))  

        # Optimizer
        #sgd = SGD(lr=learning_rate)
        adam = Adam(lr = learning_rate) # Adaptive Moment Estimation : faster than sgd

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', f1])
        return model
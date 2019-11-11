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

# import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
import math
import numpy as np


# from sklearn.metrics import f1_score

# Inspiré de : https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    # return m.update_state()


def f1(y_true, y_pred):
    # precision = #m.update_state() # precision_m(y_true, y_pred)
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Inspiré de : https://github.com/LeighWeston86/multilayer_perceptron
def RN_model(layer_sizes, dropout, learning_rate):
    '''
    Multilayer perceptron for binary classification.
    :param layer_sizes: list; size for each hidden layer
    :param dropout: float; dropout for hidden layers
    :param learning_rate: float; learning rate for Adam optmizer
    :return: keras model; compiled model for multilayer perceptron
    '''
    with tf.device('/GPU:0'):

        model = Sequential()
        model.add(Dense(77))  # couche entrée
        model.add(Activation('relu'))

        for size in layer_sizes:
            model.add(Dense(size))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))  # couche de sortie
        #sgd = SGD(lr=learning_rate)
        adam = Adam(lr = learning_rate)

        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', f1])
        return model

# def initialize_weights():
#     '''Model has 3 hidden layers of size 100, 150, 50;
#     Input layer has size 784, output layer has size 10'''
#     weights = {}

#     #1st hidden layer
#     weights['w_1'] = tf.Variable(tf.random_normal(shape = [77, 50]))
#     weights['b_1'] = tf.Variable(tf.random_normal(shape = [50]))

#     # #2nd hidden layer
#     weights['w_2'] = tf.Variable(tf.random_normal(shape = [50, 25]))
#     weights['b_2'] = tf.Variable(tf.random_normal(shape = [25]))

#     # #3rd hidden layer
#     # weights['w_3'] = tf.Variable(tf.random_normal(shape = [150, 50]))
#     # weights['b_3'] = tf.Variable(tf.random_normal(shape = [50]))

#     #output layer
#     weights['w_out'] = tf.Variable(tf.random_normal(shape = [25, 1]))
#     weights['b_out'] = tf.Variable(tf.random_normal(shape=[1]))

#     return weights

# def neural_network(X, weights):

#     #1st hidden layer
#     Z1 = tf.add(tf.matmul(X, weights['w_1']), weights['b_1'])
#     A1 = tf.nn.relu(Z1)

#     #2nd hidden layer
#     Z2 = tf.add(tf.matmul(A1, weights['w_2']), weights['b_2'])
#     A2 = tf.nn.relu(Z2)

#     #3rd hidden layer
#     # Z3 = tf.add(tf.matmul(A2, weights['w_3']), weights['b_3'])
#     # A3 = tf.nn.relu(Z3)

#     #output layer
#     out = tf.add(tf.matmul(A2, weights['w_out']), weights['b_out'])

#     return out

# def get_minibatches(X, y, batch_size):
#     num_batches = math.ceil(X.shape[1]/batch_size)
#     X_batches = np.array_split(X, num_batches, axis = 0)
#     y_batches = np.array_split(y, num_batches, axis=0)
#     return [(X_batch, y_batch) for X_batch, y_batch in zip(X_batches, y_batches)]

# def fit_model(X_train, X_test, y_train, y_test, learning_rate = 0.001, batch_size = 32, epochs = 5):
#     print(X_train.shape, y_train.shape, X_test.shape,y_test.shape)
#     print(X_train.shape[0],X_train.shape[1])
#     #Shape

#     num_features = X_train.shape[1]
#     num_classes  = 2#y_train.shape[1]

#     #Create the placeholders
#     X = tf.placeholder(dtype = tf.float32, shape = [None, num_features])
#     y = tf.placeholder(dtype = tf.float32, shape = [None, num_classes])

#     #Initialize the weights
#     weights = initialize_weights()

#     #Define the logits
#     out = neural_network(X, weights)

#     #Define the cost and optimizer)
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y))
#     optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#     #initialize
#     init = tf.global_variables_initializer()

#     #Run the session
#     with tf.Session() as sess:

#         #Initialize
#         sess.run(init)

#         for epoch in range(epochs):
#             #Get the minibatches
#             minibatches = get_minibatches(X_train, y_train, batch_size)
#             num_batches = len(minibatches)
#             epoch_cost = 0
#             for minibatch in minibatches:
#                 X_batch, y_batch = minibatch
#                 _, batch_cost = sess.run([optimizer, cost], feed_dict= {X:X_batch, y:y_batch})
#                 epoch_cost += batch_cost/num_batches
#             print(epoch, epoch_cost)

#         # Test model
#         pred = tf.nn.softmax(out)  # Apply softmax to logits
#         correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#         # Calculate accuracy
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#         print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))

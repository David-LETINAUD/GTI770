`#! /usr/bin/env python3                                                                                                      
# -*- coding: utf-8 -*-      

# Imports                                                                                                                                                 
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import MinMaxScaler

#pour les premiers tests                                                                                                                                  
from sklearn import datasets
iris = datasets.load_iris()

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target)


#scale data + multinomial bayes                                                                                                                           
scaler = MinMaxScaler()
#print(scaler.fit(data))                                                                                                                                  
scale_data = scaler.fit(iris.data, iris.target).predict(iris.data)
mnb = MultinomialNB()
print(mnb.fit(iris.data,iris.target))

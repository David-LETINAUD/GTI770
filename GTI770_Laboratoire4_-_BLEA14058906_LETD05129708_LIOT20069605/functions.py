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

import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd


########################################   Lecture   ########################################
def get_data(dataset_path):
    """
    Lit les données, normalise et découpage du dataset      
    output : 
        (np.ndarray) : X, Y
    """
    X=[]
    Y=[]

    features_list = pd.read_csv(dataset_path, header=None, sep = ',')

    Y = np.array(features_list.iloc[:,-1])    
    X = np.array(features_list.iloc[:,1:-1])

    return X, Y

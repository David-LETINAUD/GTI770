#! /usr/bin/env python3                                                                                                                                                              
# -*- coding: utf-8 -*-                                                                                                                                                              

"""                                                                                                                                                                                  
Course :                                                                                                                                                                             
GTI770 — Systèmes intelligents et apprentissage machine                                                                                                                              
                                                                                                                                                                                     
Project :                                                                                                                                                                            
Lab # 4 — Développement d'un système intelligent                                                                                                                         
                                                                                                                                                                                     
Students :                                                                                                                                                                           
Alexendre Bleau — BLEA14058906                                                                                                                                                       
David Létinaud  — LETD05129708                                                                                                                                                       
Thomas Lioret   — LIOT20069605                                                                                                                                                       
                                                                                                                                                                                     
Group :                                                                                                                                                                              
GTI770-A19-01                                                                                                                                                                        
"""

#Imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import sklearn.metrics as metrics


#fonctions


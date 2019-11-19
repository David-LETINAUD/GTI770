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

from functions import get_data

MFCC_path = "./tagged_feature_sets/msd-jmirmfccs_dev/msd-jmirmfccs_dev.csv"

X, Y = get_data(MFCC_path)





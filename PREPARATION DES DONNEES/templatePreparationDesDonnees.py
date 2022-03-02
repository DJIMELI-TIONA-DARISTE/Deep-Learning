#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *****************MY TEMPLATE****************************
 #Chargement des packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#chargement des données
dataset=pd.read_csv("Data.csv")

#Tableau des variables indépendate et des variable dépendante
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Séparation du jeux de données: partie entrainement & partie test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#changement d'échelle
from sklearn.preprocessing import  StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

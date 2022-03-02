#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 #Preparation des données 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#chargement des données
dataset=pd.read_csv("Data.csv")

#Tableau des variables indépendate et des variable dépendante
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

#Traitement des données manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:, 1:3] = imputer.transform(x[:,1:3])

#Encodage des variables catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
       #encodage de la variable country 
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:, 0])
ct = ColumnTransformer([("ountry", OneHotEncoder(), [0])], remainder = 'passthrough')
x = ct.fit_transform(x)
      # encodage de la variable purchased
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
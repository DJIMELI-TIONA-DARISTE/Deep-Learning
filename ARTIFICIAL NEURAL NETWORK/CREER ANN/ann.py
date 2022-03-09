# ARTIFICIAL NEURAL NETWORK

# Partie 1 : PREPARATION DES DONNEES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#chargement des données
dataset=pd.read_csv("Churn_Modelling.csv")

#Tableau des variables indépendate et des variable dépendante
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#ENCODAGE DES VARIABLES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:, 1])
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:, 2])
ct1 = ColumnTransformer([("Geography", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct1.fit_transform(x)
x =x[:, 1:]

#Séparation du jeux de données: partie entrainement & partie test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                    test_size=0.2, random_state=0)

#changement d'échelle
from sklearn.preprocessing import  StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Partie 2: CONSTRUCTION DU RESEAU DE NEURONE

#Importation du module de keras 
import keras
from keras.models import Sequential
from keras.layers import Dense

#INITIALISATION DU RESEAU DE NEURONE
classifier = Sequential()

#AJOUTER UNE COUCHE D ENTREE ET UNE COUCHE CACHEE
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform", 
                     input_dim=11))

# AJOUTER UNE COUCHE CACHEE
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform"))




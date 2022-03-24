#Importation des modules
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D 
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#PARTIE 1: PREPARATION DES DONNEES

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                   'dataset\\training_set',
                                                    target_size=(64, 64 ), 
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                             'dataset\\test_set',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')
#partie 2: construction du CNN
#Initialiser le CNN
classifier = Sequential()

#Etape 1 : Convolution 
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,
                            input_shape=(64, 64, 3), activation="relu" ))

#Etape 2 : MaxPooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Ajout d'une couche de convolution 
classifier.add(Convolution2D(filters=32,kernel_size=3,strides=1,
                             activation="relu" ))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#Etape 3 : Flattening
classifier.add(Flatten())

#Etape 4 : Couche completement connectee
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

#Compilation  
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

# Entrainer le CNN  sur nos images

classifier.fit_generator(
                    training_set,
                    samples_per_epoch=250,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=63)

#training accuracy = 0.7034
#Test accuracy = 0.6918
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("dataset\\single_prediction\\cat_or_dog_2.jpg",
                            target_size=(64, 64 ))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
resultat = classifier.predict(test_image)
training_set.class_indices

if resultat[0][0]==1:
    prediction = "chien"
else:
    prediction = "chat"
print(prediction)



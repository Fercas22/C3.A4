from pickletools import optimize
from tabnanny import verbose
import zipfile
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

def descomprimirArchivos():
    trainZip = './zips/train.zip'
    zip_ref = zipfile.ZipFile(trainZip, 'r')
    zip_ref.extractall('./temp')
    zip_ref.close()

    testZip = './zips/test.zip'
    zip_ref = zipfile.ZipFile(testZip, 'r')
    zip_ref.extractall('./temp')
    zip_ref.close()

def imagenesClasificadasBlancoYNegro():
    TRAININ_DIR = './temp/train'

    trainin_datagen = ImageDataGenerator(rescale=1. / 255)
    
    trainin_generator = trainin_datagen.flow_from_directory(
        TRAININ_DIR,
        #color_mode='grayscale',
        target_size=(150, 150),
        class_mode='categorical',
        batch_size= 8
    )

    VALIDATION_DIR = './temp/test'

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        #color_mode='grayscale',
        target_size=(150, 150),
        class_mode='categorical',
        batch_size= 8
    )

    return [trainin_generator, validation_generator]

def crearModeloConvulucional():

    imagenes = imagenesClasificadasBlancoYNegro()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(50, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(50, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(100, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(100, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(500, activation='relu'),

        #SALIDA 
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

   
    history = model.fit_generator(
        imagenes[0],
        epochs=25,
        validation_data= imagenes[1],
        verbose=1)
    
    model.save('./models/modelo.h5')
    model.save_weights('./models/pesos.h5')
    
    loss = history.history['loss']

    plt.xlabel('# Epoca')
    plt.ylabel('Magnitud de perdida')
    plt.plot(loss)

    plt.show()

if __name__ == '__main__':
    descomprimirArchivos()
    print('-----Convulucional------')
    crearModeloConvulucional()
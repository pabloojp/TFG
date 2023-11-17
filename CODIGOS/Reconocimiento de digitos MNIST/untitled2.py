# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 19:17:42 2023

@author: pjime
"""

"""
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.
"""

from PIL import Image
import csv
import pandas as pd                         # Pandas nos permite analizar datos grandes y obtener conclusiones basadas en teorías estadísticas. Pandas puede limpiar conjuntos de datos desordenados y hacer que sean legibles y relevantes.
import numpy as np                          # NumPy tiene como objetivo proporcionar un objeto de matriz que es hasta 50 veces más rápido que las listas tradicionales de Python.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg            # Son dos módulos de la biblioteca Matplotlib, que se utilizan para crear y mostrar gráficos, imágenes y visualizaciones en Python
'''import seaborn as sns                       # Seaborn es una biblioteca para crear gráficos estadísticos en Python. Se basa en matplotlib e integra estrechamente las estructuras de datos de pandas. Seaborn te ayuda a explorar y comprender tus datos

from sklearn.model_selection import train_test_split   # La función train_test_split se utiliza para dividir un conjunto de datos en dos subconjuntos: uno para entrenamiento del modelo y otro para evaluación.
from sklearn.metrics import confusion_matrix           # Importante para evaluar el rendimiento de un modelo de clasificación.
import itertools                                       # itertools es un módulo que proporciona herramientas de iteración eficientes y versátiles en Python, inspiradas en conceptos de otros lenguajes, lo que permite realizar operaciones avanzadas de iteración y manipulación de datos de manera concisa y eficiente.

import tensorflow as tf 
from keras.utils import to_categorical                                # Se importa la clase to_categorical, que realiza la codificación one-hot de las etiquetas en problemas de clasificación multiclase. La codificación one-hot es una técnica que convierte etiquetas categóricas en una representación numérica que es más adecuada para su uso en algoritmos de aprendizaje automático, especialmente en redes neuronales.
from keras.models import Sequential                                   # Se importa la clase Sequential de keras.models. Sequential es un tipo de modelo en el que las capas se apilan una encima de la otra en una secuencia lineal.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # Se importan varias capas de redes neuronales desde el módulo layers de Keras. Estas capas son componentes fundamentales para construir modelos de redes neuronales convolucionales (CNN). Incluyen capas densas (Dense), capas de eliminación (Dropout), capas de aplanamiento (Flatten), capas convolucionales (Conv2D) y capas de agrupación máxima (MaxPool2D).
from keras.optimizers import RMSprop                                  # Se importa el optimizador RMSprop, que es un algoritmo de optimización utilizado para ajustar los pesos y los sesgos de una red neuronal durante el entrenamiento.
from keras.preprocessing.image import ImageDataGenerator              # Se importa ImageDataGenerator, que es una clase que se utiliza para realizar la generación de imágenes aumentadas y realizar la preprocesamiento de imágenes para su uso en el entrenamiento de modelos de redes neuronales. Esto es útil para aumentar la cantidad de datos de entrenamiento y mejorar la capacidad del modelo para generalizar.
from keras.callbacks import ReduceLROnPlateau                         # Se importa ReduceLROnPlateau, que es una devolución de llamada utilizada para reducir la tasa de aprendizaje durante el entrenamiento si ciertas condiciones no se cumplen.


np.random.seed(2)                                    # Al poner esto, haces que se generen datos aleatorios pero cada vez que reproduzcas este codigo aparecen los mismos datos aleatorios entre [0,1)
'''

# Cargar los datos de entrenamiento y prueba
def cargar_datos(train,test):
    train = csv.reader(train)
    test = csv.reader(test)

    Y_train = train.iloc[:, 0]
    X_train = train.iloc[:,[1,len(train)-1]]

    
    return len(train)



# Chequear si hay alguno nulo


# Normalizar los datos


# Reshape


# to_categorical


#train_test_split


# crear el modelo


#optimizador y compilar el modelo


#learning rate annealer


#epocas y batch_size


# datagen


#entrenar el modelo


# matriz de confusion
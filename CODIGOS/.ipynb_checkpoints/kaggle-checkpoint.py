#Código sacado de https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook


#Importación de bibliotecas:
import pandas as pd     # Pandas nos permite analizar datos grandes y obtener conclusiones basadas en teorías estadísticas. Pandas puede limpiar conjuntos de datos desordenados y hacer que sean legibles y relevantes.
import numpy as np      # NumPy tiene como objetivo proporcionar un objeto de matriz que es hasta 50 veces más rápido que las listas tradicionales de Python.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg   # Son dos módulos de la biblioteca Matplotlib, que se utilizan para crear y mostrar gráficos, imágenes y visualizaciones en Python
import seaborn as sns   # Seaborn es una biblioteca para crear gráficos estadísticos en Python. Se basa en matplotlib e integra estrechamente las estructuras de datos de pandas. Seaborn te ayuda a explorar y comprender tus datos


# Configuración de generador de números aleatorios:
np.random.seed(2)       # Al poner esto, haces que se generen datos aleatorios pero cada vez que reproduzcas este codigo aparecen los mismos datos aleatorios entre [0,1)


# Importo funciones y clases relacionadas con el aprendizaje automático.
from sklearn.model_selection import train_test_split   # La función train_test_split se utiliza para dividir un conjunto de datos en dos subconjuntos: uno para entrenamiento del modelo y otro para evaluación.
from sklearn.metrics import confusion_matrix           # Importante para evaluar el rendimiento de un modelo de clasificación.
import itertools                                       # itertools es un módulo que proporciona herramientas de iteración eficientes y versátiles en Python, inspiradas en conceptos de otros lenguajes, lo que permite realizar operaciones avanzadas de iteración y manipulación de datos de manera concisa y eficiente.


# Importo de algunas componentes de Keras.
from tensorflow.keras.utils import to_categorical                     # Se importa la clase to_categorical, que realiza la codificación one-hot de las etiquetas en problemas de clasificación multiclase. La codificación one-hot es una técnica que convierte etiquetas categóricas en una representación numérica que es más adecuada para su uso en algoritmos de aprendizaje automático, especialmente en redes neuronales.
from keras.models import Sequential                                   # Se importa la clase Sequential de keras.models. Sequential es un tipo de modelo en el que las capas se apilan una encima de la otra en una secuencia lineal.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # Se importan varias capas de redes neuronales desde el módulo layers de Keras. Estas capas son componentes fundamentales para construir modelos de redes neuronales convolucionales (CNN). Incluyen capas densas (Dense), capas de eliminación (Dropout), capas de aplanamiento (Flatten), capas convolucionales (Conv2D) y capas de agrupación máxima (MaxPool2D).
from keras.optimizers import RMSprop                                  # Se importa el optimizador RMSprop, que es un algoritmo de optimización utilizado para ajustar los pesos y los sesgos de una red neuronal durante el entrenamiento.
from keras.preprocessing.image import ImageDataGenerator              # Se importa ImageDataGenerator, que es una clase que se utiliza para realizar la generación de imágenes aumentadas y realizar la preprocesamiento de imágenes para su uso en el entrenamiento de modelos de redes neuronales. Esto es útil para aumentar la cantidad de datos de entrenamiento y mejorar la capacidad del modelo para generalizar.
from keras.callbacks import ReduceLROnPlateau                         # Se importa ReduceLROnPlateau, que es una devolución de llamada utilizada para reducir la tasa de aprendizaje durante el entrenamiento si ciertas condiciones no se cumplen.


# Configuramos el estilo Seaborn
sns.set(style='white', context='paper', palette='deep')     # Configura las preferencias de estilo de Seaborn


# Cargamos los datos de tipo CSV que he descargado en la web Kaggle
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# Inicializamos las variables Y_train y X_train.
Y_train = train["label"]                           # Inicializamos la variable Y_train que contiene la primera columna de los datos introducidos en train.csv. En ella se almacena el valor del numero en cada ejemplo.
X_train = train.drop(labels = ["label"],axis = 1)  # Caracteristicas de entrada de cada ejemplo.


# Para eliminar espacio, como ya tenemos la información relevante en Y_train y x_train, eliminamos train.
del train 


# Muestra la distribución de las etiquetas de cada tipo en un grafico.
g = sns.countplot(Y_train)


# Calcula el recuento de cada clase
Y_train.value_counts()




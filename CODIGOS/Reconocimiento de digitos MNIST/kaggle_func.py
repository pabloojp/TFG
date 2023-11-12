"""
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento y de prueba como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.
"""

#Importación de bibliotecas:
import pandas as pd     # Pandas nos permite analizar datos grandes y obtener conclusiones basadas en teorías estadísticas. Pandas puede limpiar conjuntos de datos desordenados y hacer que sean legibles y relevantes.
import numpy as np      # NumPy tiene como objetivo proporcionar un objeto de matriz que es hasta 50 veces más rápido que las listas tradicionales de Python.
from sklearn.model_selection import train_test_split   # La función train_test_split se utiliza para dividir un conjunto de datos en dos subconjuntos: uno para entrenamiento del modelo y otro para evaluación.
import itertools                                       # itertools es un módulo que proporciona herramientas de iteración eficientes y versátiles en Python, inspiradas en conceptos de otros lenguajes, lo que permite realizar operaciones avanzadas de iteración y manipulación de datos de manera concisa y eficiente.
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical                     # Se importa la clase to_categorical, que realiza la codificación one-hot de las etiquetas en problemas de clasificación multiclase. La codificación one-hot es una técnica que convierte etiquetas categóricas en una representación numérica que es más adecuada para su uso en algoritmos de aprendizaje automático, especialmente en redes neuronales.
from keras.models import Sequential                                   # Se importa la clase Sequential de keras.models. Sequential es un tipo de modelo en el que las capas se apilan una encima de la otra en una secuencia lineal.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # Se importan varias capas de redes neuronales desde el módulo layers de Keras. Estas capas son componentes fundamentales para construir modelos de redes neuronales convolucionales (CNN). Incluyen capas densas (Dense), capas de eliminación (Dropout), capas de aplanamiento (Flatten), capas convolucionales (Conv2D) y capas de agrupación máxima (MaxPool2D).
from keras.optimizers import RMSprop                                  # Se importa el optimizador RMSprop, que es un algoritmo de optimización utilizado para ajustar los pesos y los sesgos de una red neuronal durante el entrenamiento.
from keras.preprocessing.image import ImageDataGenerator              # Se importa ImageDataGenerator, que es una clase que se utiliza para realizar la generación de imágenes aumentadas y realizar la preprocesamiento de imágenes para su uso en el entrenamiento de modelos de redes neuronales. Esto es útil para aumentar la cantidad de datos de entrenamiento y mejorar la capacidad del modelo para generalizar.
from keras.callbacks import ReduceLROnPlateau                         # Se importa ReduceLROnPlateau, que es una devolución de llamada utilizada para reducir la tasa de aprendizaje durante el entrenamiento si ciertas condiciones no se cumplen.

np.random.seed(2)       # Al poner esto, haces que se generen datos aleatorios pero cada vez que reproduzcas este codigo aparecen los mismos datos aleatorios entre [0,1)


# Función para cargar los datos
def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    Y_train = train["label"]                         # Inicializamos la variable Y_train que contiene la primera columna de los datos introducidos en train.csv. En ella se almacena el numero correspondiente (etiquetas) a cada ejemplo de train.
    X_train = train.drop(labels=["label"], axis=1)   # Caracteristicas de entrada de cada ejemplo sin las etiquetas de cada ejemplo.
    del train

    X_train = X_train / 255.0                        # Normalizamos los datos para regular la iluminacion (los pixeles tienen valores de 0 a 255) porque las CNN convergen más rápidamente con datos en el rango [0..1]
    test = test / 255.0

    X_train = X_train.values.reshape(-1, 28, 28, 1)  # Las imágenes de entrenamiento y prueba, que tienen un tamaño de 28 píxeles x 28 píxeles, se han almacenado en un DataFrame de pandas como vectores unidimensionales de 784 valores. Este paso, nos devuelve los datos como matrices 28x28x1 (1 canal, escala grises)
    test = test.values.reshape(-1, 28, 28, 1)

    Y_train = to_categorical(Y_train, num_classes=10) # Codificamos las etiquetas en vectores "one-hot". Cada etiqueta la representamos como un vector binario en el que un solo elemento tiene un valor de 1 y los demás son 0.

    return X_train, Y_train, test


# Función para dividir los datos en entrenamiento y validación
def split_data(X_train, Y_train, test_size=0.1, random_seed=2):       # Establecemos la semilla aleatoria en 2. Dividimos el conjunto de entrenamiento en dos partes: la validación (10%) y el entreno (90%), para asegurar que el modelo pueda ser evaluado en datos que no ha visto durante el entrenamiento. La división aleatoria garantiza que no haya un desequilibrio en las etiquetas entre el conjunto de entrenamiento y el conjunto de validación. 

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=test_size, random_state=random_seed)
    return X_train, X_val, Y_train, Y_val


# Función para definir el modelo CNN. La arquitectura es : In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
def create_model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model


# Función para compilar el modelo
def compile_model(model):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, centered=False)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# Función para configurar el generador de imágenes aumentadas
def configure_image_generator(X_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(X_train)
    return datagen


# Función para entrenar el modelo
def train_model(model, datagen, X_train, Y_train, X_val, Y_val, epochs=5, batch_size=86):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, Y_val), verbose=2, steps_per_epoch=X_train.shape[0] // batch_size, callbacks=[learning_rate_reduction])
    model.save('modelo_digitos_func.keras')


if __name__ == "__main__":
    # Cargar datos
    X_train, Y_train, test = load_data("train.csv", "test.csv")      # 42000 ejemplos de entrenamiento con 784 pixeles (imagenes 28x28). En train, las dimensiones son 42001 x 785 y 28000 ejemplos de test con 784 pixeles.

    # Dividir datos
    X_train, X_val, Y_train, Y_val = split_data(X_train, Y_train)

    # Crear modelo
    model = create_model()

    # Compilar modelo
    compile_model(model)

    # Configurar generador de imágenes
    datagen = configure_image_generator(X_train)

    # Entrenar modelo
    train_model(model, datagen, X_train, Y_train, X_val, Y_val, epochs=10, batch_size=86)


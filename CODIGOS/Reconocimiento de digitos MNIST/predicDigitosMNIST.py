"""
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.
"""

#Importación de bibliotecas:
import pandas as pd     # Pandas nos permite analizar datos grandes y obtener conclusiones basadas en teorías estadísticas. Pandas puede limpiar conjuntos de datos desordenados y hacer que sean legibles y relevantes.
import numpy as np      # NumPy tiene como objetivo proporcionar un objeto de matriz que es hasta 50 veces más rápido que las listas tradicionales de Python.
from sklearn.model_selection import train_test_split   # La función train_test_split se utiliza para dividir un conjunto de datos en dos subconjuntos: uno para entrenamiento del modelo y otro para evaluación.
import itertools                                       # itertools es un módulo que proporciona herramientas de iteración eficientes y versátiles en Python, inspiradas en conceptos de otros lenguajes, lo que permite realizar operaciones avanzadas de iteración y manipulación de datos de manera concisa y eficiente.
import tensorflow as tf 
from keras.utils import to_categorical                                # Se importa la clase to_categorical, que realiza la codificación one-hot de las etiquetas en problemas de clasificación multiclase. La codificación one-hot es una técnica que convierte etiquetas categóricas en una representación numérica que es más adecuada para su uso en algoritmos de aprendizaje automático, especialmente en redes neuronales.
from keras.models import Sequential                                   # Se importa la clase Sequential de keras.models. Sequential es un tipo de modelo en el que las capas se apilan una encima de la otra en una secuencia lineal.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # Se importan varias capas de redes neuronales desde el módulo layers de Keras. Estas capas son componentes fundamentales para construir modelos de redes neuronales convolucionales (CNN). Incluyen capas densas (Dense), capas de eliminación (Dropout), capas de aplanamiento (Flatten), capas convolucionales (Conv2D) y capas de agrupación máxima (MaxPool2D).
from keras.optimizers import RMSprop                                  # Se importa el optimizador RMSprop, que es un algoritmo de optimización utilizado para ajustar los pesos y los sesgos de una red neuronal durante el entrenamiento.
from keras.preprocessing.image import ImageDataGenerator              # Se importa ImageDataGenerator, que es una clase que se utiliza para realizar la generación de imágenes aumentadas y realizar la preprocesamiento de imágenes para su uso en el entrenamiento de modelos de redes neuronales. Esto es útil para aumentar la cantidad de datos de entrenamiento y mejorar la capacidad del modelo para generalizar.
from keras.callbacks import ReduceLROnPlateau                         # Se importa ReduceLROnPlateau, que es una devolución de llamada utilizada para reducir la tasa de aprendizaje durante el entrenamiento si ciertas condiciones no se cumplen.

np.random.seed(2)                                    # Al poner esto, haces que se generen datos aleatorios pero cada vez que reproduzcas este codigo aparecen los mismos datos aleatorios entre [0,1)


# Función para cargar los datos
def load_data(train_file, test_file):

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    Y_train = train["label"]                         # Inicializamos la variable Y_train que contiene la primera columna de los datos introducidos en train.csv. En ella se almacena el numero correspondiente (etiquetas) a cada ejemplo de train.
    X_train = train.drop(labels=["label"], axis=1)   # Caracteristicas de entrada de cada ejemplo sin las etiquetas de cada ejemplo.
    del train


    #Normalizamos los datos
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
    # Filtros de 32 canales: Los filtros son pequeñas matrices que se utilizan para escanear partes de la imagen de entrada. En este caso, se están utilizando 32 filtros en ambas capas. Cada filtro es como una "lente" que busca patrones específicos en la imagen.
    # Kernel de convolución 5x5: El kernel de convolución es una matriz que se desliza sobre la imagen para realizar una operación matemática llamada convolución. En este caso, se está utilizando un kernel de 5x5, lo que significa que cada filtro analiza una región de 5x5 píxeles de la imagen en cada paso.
    # Activación 'relu': 'relu' es una función de activación llamada Rectified Linear Unit. Después de que cada filtro haya realizado la convolución con la región de la imagen, se aplica la función 'relu'. Esta función es muy simple: si el valor de salida de la convolución es positivo, se mantiene tal cual; si es negativo, se establece en cero. Esto introduce no linealidad en la red y permite que la CNN aprenda características más complejas y patrones en los datos.
    # Entonces, en estas dos capas, cada uno de los 32 filtros busca patrones en regiones de 5x5 píxeles de la imagen de entrada y aplica la función 'relu' para resaltar las características relevantes. Estas operaciones se realizan para cada filtro, lo que permite que la CNN extraiga múltiples características de la imagen.
    # Capa MaxPool2D: La capa MaxPool2D es una capa de submuestreo que reduce la dimensionalidad de las características extraídas por las capas de convolución. En esta configuración, se utiliza una ventana de 2x2 píxeles (pool_size=(2,2)). Lo que hace esta capa es examinar grupos de 2x2 píxeles en las características de la capa anterior y retiene solo el valor máximo de esos 4 píxeles. Este proceso reduce la cantidad de información y cálculos en la red, lo que ayuda a evitar el sobreajuste y mejora el rendimiento computacional.
    # Capa Dropout: La capa Dropout es una técnica de regularización. La regularización es una forma de prevenir el sobreajuste en la red. En esta capa, se establece una fracción de las unidades de la capa anterior en cero de manera aleatoria durante el entrenamiento. En este caso, el valor es 0.25, lo que significa que aproximadamente el 25% de las unidades de esta capa se establecerán en cero en cada paso de entrenamiento. Esto fuerza a la red a aprender de manera más robusta y generalizable, ya que no puede depender demasiado de ninguna unidad específica.

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())                          #  Esta capa toma las características generadas por las capas de convolución y las convierte en un vector unidimensional.
    model.add(Dense(256, activation="relu"))      # Se agrega una capa con 256 neuronas y activacion relu. Esta capa es una capa completamente conectada y se utiliza para aprender patrones más globales de las características extraídas por las capas de convolución.
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))    # Esta es la capa de salida que produce la distribución de probabilidad de las 10 clases posibles.
    
    model.summary()                               # Imprimimos un resumen de la arquitectura del modelo

    return model


# Función para compilar el modelo
def compile_model(model):
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, centered=False)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# Función para configurar el generador de imágenes aumentadas
def configure_image_generator(X_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,             # establecer la media de entrada en 0 sobre el conjunto de datos
        samplewise_center=False,              # establecer la media de cada muestra en 0
        featurewise_std_normalization=False,  # dividir las entradas por la desviación estándar del conjunto de datos
        samplewise_std_normalization=False,   # dividir cada entrada por su desviación estándar
        zca_whitening=False,                  # aplicar blanqueo ZCA
        rotation_range=10,                    # rotar aleatoriamente las imágenes en el rango (grados, 0 a 180)
        zoom_range = 0.1,                     # Aleatoriamente hacer zoom en la imagen
        width_shift_range=0.1,                # desplazar aleatoriamente las imágenes horizontalmente (fracción del ancho total)
        height_shift_range=0.1,               # desplazar aleatoriamente las imágenes verticalmente (fracción de la altura total)
        horizontal_flip=False,                # voltear aleatoriamente las imágenes horizontalmente
        vertical_flip=False)                  # voltear aleatoriamente las imágenes verticalmente

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
    train_model(model, datagen, X_train, Y_train, X_val, Y_val, epochs=20, batch_size=86)

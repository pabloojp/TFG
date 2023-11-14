"""
Nombre del codigo: Modelo de CNN de reconocimiento de señales.
Guiado por: Tutorial de Kaggle (acceso al enlace el 12 de noviembre)
        https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn#6.-Training-the-Model
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme las imagenes de entrenamiento en diferentes carpetas (cada clase en una carpeta) 
y guardarlas en la misma carpeta donde estaba este script.
"""


# Importamos las bibliotecas necesarias
import numpy as np                                     # Importamos la biblioteca NumPy para operaciones numéricas eficientes
import pandas as pd                                    # Importamos la biblioteca pandas para manipulación y análisis de datos
import tensorflow as tf                                # Importamos TensorFlow, una biblioteca para aprendizaje automático
import os                                              # Importamos el módulo os para interactuar con el sistema operativo
from PIL import Image                                  # Importamos la clase Image del módulo PIL para trabajar con imágenes
from sklearn.model_selection import train_test_split   # Importamos train_test_split para dividir los datos en conjuntos de entrenamiento y prueba
from keras.utils import to_categorical                 # Importamos to_categorical para codificar las etiquetas en formato one-hot
from keras.models import Sequential                    # Importamos Sequential, un modelo lineal para apilar capas de red neuronal
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Importamos capas específicas para construir una red neuronal convolucional


# Definimos una función para cargar y procesar datos
def load_and_preprocess_data(data_directory):
    data = []                                         # Lista para almacenar datos de imágenes
    labels = []                                       # Lista para almacenar etiquetas de imágenes
    classes = 43                                      # Número total de clases en el conjunto de datos (las 43 señales diferentes)

    # Recorremos cada clase
    for i in range(classes):
        path = os.path.join(data_directory, 'train', str(i))  # Construimos la ruta para cada clase
        images = os.listdir(path)                             # Listamos todas las imágenes en la carpeta de la clase

        # Recorremos cada imagen en la clase
        for a in images:
            try:
                image = Image.open(path + '/' + a)   # Cargamos la imagen utilizando PIL
                image = image.resize((30, 30))       # Redimensionamos la imagen a 30x30 píxeles
                image = np.array(image)              # Convertimos la imagen a un array de NumPy
                data.append(image)                   # Agregamos la imagen a la lista de datos
                labels.append(i)                     # Agregamos la etiqueta correspondiente a la lista de etiquetas
            except:
                print("Error al cargar la imagen")

    data = np.array(data)      # Convertimos la lista de datos a un array de NumPy
    labels = np.array(labels)  # Convertimos la lista de etiquetas a un array de NumPy
    return data, labels

# Definimos una función para dividir y codificar los datos
def split_and_encode_data(data, labels, test_size=0.2, random_state=42):
    # Dividimos los datos en conjuntos de entrenamiento y prueba, y codificamos las etiquetas en formato one-hot
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    y_train = to_categorical(y_train, 43)  # Codificamos en one-hot las etiquetas de entrenamiento
    y_test = to_categorical(y_test, 43)    # Codificamos en one-hot las etiquetas de prueba
    return X_train, X_test, y_train, y_test

# Definimos una función para construir el modelo de red neuronal convolucional
def build_model(input_shape):
    model = Sequential()                                                                           # Creamos un modelo secuencial

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))  # input_shape indica la forma de los datos de entrada.
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    # Filtros de 32 canales: Los filtros son pequeñas matrices que se utilizan para escanear partes de la imagen de entrada. En este caso, se están utilizando 32 filtros en ambas capas. Cada filtro es como una "lente" que busca patrones específicos en la imagen.
    # Kernel de convolución 5x5: El kernel de convolución es una matriz que se desliza sobre la imagen para realizar una operación matemática llamada convolución. En este caso, se está utilizando un kernel de 5x5, lo que significa que cada filtro analiza una región de 5x5 píxeles de la imagen en cada paso.
    # Activación 'relu': 'relu' es una función de activación llamada Rectified Linear Unit. Después de que cada filtro haya realizado la convolución con la región de la imagen, se aplica la función 'relu'. Esta función es muy simple: si el valor de salida de la convolución es positivo, se mantiene tal cual; si es negativo, se establece en cero. Esto introduce no linealidad en la red y permite que la CNN aprenda características más complejas y patrones en los datos.
    # Entonces, en estas dos capas, cada uno de los 32 filtros busca patrones en regiones de 5x5 píxeles de la imagen de entrada y aplica la función 'relu' para resaltar las características relevantes. Estas operaciones se realizan para cada filtro, lo que permite que la CNN extraiga múltiples características de la imagen.
    # Capa MaxPool2D: La capa MaxPool2D es una capa de submuestreo que reduce la dimensionalidad de las características extraídas por las capas de convolución. En esta configuración, se utiliza una ventana de 2x2 píxeles (pool_size=(2,2)). Lo que hace esta capa es examinar grupos de 2x2 píxeles en las características de la capa anterior y retiene solo el valor máximo de esos 4 píxeles. Este proceso reduce la cantidad de información y cálculos en la red, lo que ayuda a evitar el sobreajuste y mejora el rendimiento computacional.
    # Capa Dropout: La capa Dropout es una técnica de regularización. La regularización es una forma de prevenir el sobreajuste en la red. En esta capa, se establece una fracción de las unidades de la capa anterior en cero de manera aleatoria durante el entrenamiento. En este caso, el valor es 0.25, lo que significa que aproximadamente el 25% de las unidades de esta capa se establecerán en cero en cada paso de entrenamiento. Esto fuerza a la red a aprender de manera más robusta y generalizable, ya que no puede depender demasiado de ninguna unidad específica.

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
 
    model.add(Flatten())                               #  Esta capa toma las características generadas por las capas de convolución y las convierte en un vector unidimensional.
    model.add(Dense(256, activation='relu'))           # Se agrega una capa con 256 neuronas y activacion relu. Esta capa es una capa completamente conectada y se utiliza para aprender patrones más globales de las características extraídas por las capas de convolución.
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))         # Esta es la capa de salida que produce la distribución de probabilidad de las 10 clases posibles.

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()  # Imprimimos un resumen de la arquitectura del modelo
    return model

# Definimos una función para entrenar el modelo
def train_model(model, file_name, X_train, y_train, X_test, y_test, batch_size=32, epochs=1):
    with tf.device('/GPU:0'):   # Indicamos que el entrenamiento del modelo se realizará en la GPU si está disponible; si no, en la CPU
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.save(file_name)       # Guardamos el modelo entrenado en un archivo

'''
La diferencia principal entre realizar el entrenamiento del modelo en la GPU o en la CPU radica en el rendimiento y la velocidad de entrenamiento.

GPU (Unidad de Procesamiento Gráfico):

Ventajas:
Las GPU están diseñadas específicamente para manejar operaciones matriciales y paralelas, que son comunes en el entrenamiento de modelos de redes neuronales.
Pueden realizar cálculos en paralelo en grandes cantidades de datos, lo que acelera significativamente el entrenamiento de modelos, especialmente en tareas intensivas en cálculos, como las redes neuronales profundas.
Ofrecen un rendimiento superior en comparación con las CPU para tareas de aprendizaje profundo.

Desventajas:
Pueden ser más costosas y consumir más energía que las CPU.
Puede haber limitaciones en la cantidad de memoria de la GPU disponible, lo que podría ser un factor en modelos muy grandes.

CPU (Unidad Central de Procesamiento):

Ventajas:
Disponibles en la mayoría de las computadoras y servidores sin necesidad de hardware adicional.
Adecuadas para tareas generales de propósito múltiple y no solo para aprendizaje profundo.
Pueden ser más económicas en términos de hardware y consumo de energía.
Desventajas:
Las CPU no están diseñadas específicamente para tareas de aprendizaje profundo y pueden ser menos eficientes en términos de velocidad para ciertos tipos de operaciones, especialmente en modelos grandes.

'''

if __name__ == "__main__":

    # Ruta del directorio donde se encuentran los datos
    data_directory = 'C:\\Users\\pjime\\Documents\\ESTUDIOS\\GRADO MATEMATICAS\\4-CUARTO\\TFG\\GITHUB\\CODIGOS\\Reconocimiento de señales'
    
    # Cargamos los datos
    data, labels = load_and_preprocess_data(data_directory)
    
    # Dividimos los datos
    X_train, X_test, y_train, y_test = split_and_encode_data(data, labels)
    
    # Construimos el modelo de CNN
    model = build_model(X_train.shape[1:]) # La primera indica el numero que hay, las siguientes las dimensiones (que es lo que nos interesa)
    
    # Entrenamos el modelo y lo guardamos en un archivo
    train_model(model, 'traffic_classifier.keras', X_train, y_train, X_test, y_test, 32, 20)

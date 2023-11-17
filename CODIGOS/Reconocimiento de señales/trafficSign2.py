'''
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento y de prueba como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.

https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn#6.-Training-the-Model
Me he descargado los datos de esta pagina y he seguido mas o menos las instrucciones
'''


import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import time
import datetime
import random
import seaborn as sns

def date_time(x):
    if x == 1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 2:
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x == 3:
        return 'Date now: %s' % datetime.datetime.now()
    if x == 4:
        return 'Date today: %s' % datetime.date.today()

def load_and_preprocess_data(data_directory):
    data = []
    labels = []
    classes = 43

    for i in range(classes):
        path = os.path.join(data_directory, 'train', str(i))
        images = os.listdir(path)

        for a in images:
            try:
                image = Image.open(path + '/' + a)
                image = image.resize((30, 30))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except:
                print("Error loading image")

    data = np.array(data)
    labels = np.array(labels)

    # Después de cargar y preprocesar los datos
    random_index = random.randint(0, len(data) - 1)
    random_image = data[random_index]
    random_label = labels[random_index]

    # Muestra la imagen aleatoria y su etiqueta
    plt.imshow(random_image)
    plt.title(f'Label: {random_label}')
    plt.show()

    return data, labels

def split_and_encode_data(data, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary() # Resumen del modelo
    return model

def train_model(model, file_name, X_train, y_train, X_test, y_test, batch_size=32, epochs=1):
    with tf.device('/GPU:0'):   # Indica que el entrenamiento del modelo de red neuronal se realizará en la GPU si está disponible. Sino en la CPU.
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    model.save(file_name)


if __name__ == "__main__":
    data_directory = 'C:\\Users\\pjime\\Documents\\ESTUDIOS\\GRADO MATEMATICAS\\4-CUARTO\\TFG\\GITHUB\\CODIGOS\\Reconocimiento de señales'
    data, labels = load_and_preprocess_data(data_directory)


    X_train, X_test, y_train, y_test = split_and_encode_data(data, labels)

    model = load_model('traffic_classifier.keras')

    # Calcula la matriz de confusión
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    clases = { 
            0: 'Límite de velocidad (20 km/h)',
            1: 'Límite de velocidad (30 km/h)', 
            2: 'Límite de velocidad (50 km/h)', 
            3: 'Límite de velocidad (60 km/h)', 
            4: 'Límite de velocidad (70 km/h)', 
            5: 'Límite de velocidad (80 km/h)', 
            6: 'Fin del límite de velocidad (80 km/h)', 
            7: 'Límite de velocidad (100 km/h)', 
            8: 'Límite de velocidad (120 km/h)', 
            9: 'Prohibido adelantar', 
            10: 'Prohibido adelantar vehículos de más de 3.5 toneladas', 
            11: 'Derecho de paso en intersección', 
            12: 'Carretera con prioridad', 
            13: 'Ceder el paso', 
            14: 'Detenerse', 
            15: 'Prohibido el paso de vehículos', 
            16: 'Prohibido el paso de vehículos de más de 3.5 toneladas',
            17: 'Prohibido el acceso', 
            18: 'Precaución general', 
            19: 'Curva peligrosa a la izquierda', 
            20: 'Curva peligrosa a la derecha', 
            21: 'Curva doble', 
            22: 'Carretera con baches', 
            23: 'Carretera resbaladiza', 
            24: 'Carretera se estrecha a la derecha', 
            25: 'Trabajo en la carretera', 
            26: 'Señales de tráfico', 
            27: 'Peatones', 
            28: 'Cruce de niños', 
            29: 'Cruce de bicicletas', 
            30: 'Precaución: hielo/nieve',
            31: 'Cruce de animales salvajes', 
            32: 'Fin de límites de velocidad y adelantamiento', 
            33: 'Girar a la derecha', 
            34: 'Girar a la izquierda', 
            35: 'Solo adelante', 
            36: 'Ir recto o girar a la derecha', 
            37: 'Ir recto o girar a la izquierda', 
            38: 'Mantenerse a la derecha', 
            39: 'Mantenerse a la izquierda', 
            40: 'Circulación obligatoria en rotonda', 
            41: 'Fin de la prohibición de adelantar', 
            42: 'Fin de la prohibición de adelantar vehículos de más de 3.5 toneladas'
        }

    # Cargar el modelo previamente entrenado
    model = load_model('traffic_classifier.keras')

    # Cargar la imagen
    image_path = '31.png'
    image = Image.open(image_path)
    image = image.resize((30, 30))
    image = image.convert('RGB')
    image = np.array(image)
    image = image / 255.0  # Normalizar los valores de píxeles

    # Realizar la predicción con el modelo cargado
    prediction = model.predict(np.array([image]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)

    # Obtener la etiqueta predicha (índice de la clase con mayor probabilidad)
    predicted_class = np.argmax(prediction)

    # Obtener la etiqueta correspondiente a la clase predicha
    predicted_label = clases[predicted_class]

    # Imprimir la etiqueta predicha
    print(f'Clase predicha: {predicted_label}')

    


'''  
    # Calcular la cantidad de datos por etiqueta
    label_counts = np.bincount(labels)

    # Crear un gráfico de barras con etiquetas
    plt.bar(range(len(label_counts)), label_counts, tick_label=range(len(label_counts)))

    # Agregar etiquetas con los números en cada barra
    for i, count in enumerate(label_counts):
        plt.text(i, count + 0.1, str(count), ha='center', va='bottom')

    plt.xlabel('Etiquetas')
    plt.ylabel('Cantidad de Datos')
    plt.title('Cantidad de Datos por Etiqueta')
    plt.show()


    # Grafico numero de imagenes por etiqueta en orden.
    train_path = 'C:\\Users\\pjime\\Documents\\ESTUDIOS\\GRADO MATEMATICAS\\4-CUARTO\\TFG\\GITHUB\\CODIGOS\\Reconocimiento de señales\\Train'
    classes = { 
            0: 'Límite de velocidad (20 km/h)',
            1: 'Límite de velocidad (30 km/h)', 
            2: 'Límite de velocidad (50 km/h)', 
            3: 'Límite de velocidad (60 km/h)', 
            4: 'Límite de velocidad (70 km/h)', 
            5: 'Límite de velocidad (80 km/h)', 
            6: 'Fin del límite de velocidad (80 km/h)', 
            7: 'Límite de velocidad (100 km/h)', 
            8: 'Límite de velocidad (120 km/h)', 
            9: 'Prohibido adelantar', 
            10: 'Prohibido adelantar vehículos de más de 3.5 toneladas', 
            11: 'Derecho de paso en intersección', 
            12: 'Carretera con prioridad', 
            13: 'Ceder el paso', 
            14: 'Detenerse', 
            15: 'Prohibido el paso de vehículos', 
            16: 'Prohibido el paso de vehículos de más de 3.5 toneladas',
            17: 'Prohibido el acceso', 
            18: 'Precaución general', 
            19: 'Curva peligrosa a la izquierda', 
            20: 'Curva peligrosa a la derecha', 
            21: 'Curva doble', 
            22: 'Carretera con baches', 
            23: 'Carretera resbaladiza', 
            24: 'Carretera se estrecha a la derecha', 
            25: 'Trabajo en la carretera', 
            26: 'Señales de tráfico', 
            27: 'Peatones', 
            28: 'Cruce de niños', 
            29: 'Cruce de bicicletas', 
            30: 'Precaución: hielo/nieve',
            31: 'Cruce de animales salvajes', 
            32: 'Fin de límites de velocidad y adelantamiento', 
            33: 'Girar a la derecha', 
            34: 'Girar a la izquierda', 
            35: 'Solo adelante', 
            36: 'Ir recto o girar a la derecha', 
            37: 'Ir recto o girar a la izquierda', 
            38: 'Mantenerse a la derecha', 
            39: 'Mantenerse a la izquierda', 
            40: 'Circulación obligatoria en rotonda', 
            41: 'Fin de la prohibición de adelantar', 
            42: 'Fin de la prohibición de adelantar vehículos de más de 3.5 toneladas'
        }
    
    folders = os.listdir('C:\\Users\\pjime\\Documents\\ESTUDIOS\\GRADO MATEMATICAS\\4-CUARTO\\TFG\\GITHUB\\CODIGOS\\Reconocimiento de señales\\Train')

    train_number = []
    class_num = []

    for folder in folders:
        train_files = os.listdir(train_path + '/' + folder)
        train_number.append(len(train_files))
        class_num.append(classes[int(folder)])

    # Sorting the dataset on the basis of number of images in each class
    zipped_lists = zip(train_number, class_num)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    train_number, class_num = [ list(tuple) for tuple in  tuples]

    # Plotting the number of images in each class
    plt.figure(figsize=(21,10))  
    plt.bar(class_num, train_number)
    plt.xticks(class_num, rotation='vertical')
    plt.show()



# Calcula la matriz de confusión
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    clases = { 
        0: 'Límite de velocidad (20 km/h)',
        1: 'Límite de velocidad (30 km/h)', 
        2: 'Límite de velocidad (50 km/h)', 
        3: 'Límite de velocidad (60 km/h)', 
        4: 'Límite de velocidad (70 km/h)', 
        5: 'Límite de velocidad (80 km/h)', 
        6: 'Fin del límite de velocidad (80 km/h)', 
        7: 'Límite de velocidad (100 km/h)', 
        8: 'Límite de velocidad (120 km/h)', 
        9: 'Prohibido adelantar', 
        10: 'Prohibido adelantar vehículos de más de 3.5 toneladas', 
        11: 'Derecho de paso en intersección', 
        12: 'Carretera con prioridad', 
        13: 'Ceder el paso', 
        14: 'Detenerse', 
        15: 'Prohibido el paso de vehículos', 
        16: 'Prohibido el paso de vehículos de más de 3.5 toneladas',
        17: 'Prohibido el acceso', 
        18: 'Precaución general', 
        19: 'Curva peligrosa a la izquierda', 
        20: 'Curva peligrosa a la derecha', 
        21: 'Curva doble', 
        22: 'Carretera con baches', 
        23: 'Carretera resbaladiza', 
        24: 'Carretera se estrecha a la derecha', 
        25: 'Trabajo en la carretera', 
        26: 'Señales de tráfico', 
        27: 'Peatones', 
        28: 'Cruce de niños', 
        29: 'Cruce de bicicletas', 
        30: 'Precaución: hielo/nieve',
        31: 'Cruce de animales salvajes', 
        32: 'Fin de límites de velocidad y adelantamiento', 
        33: 'Girar a la derecha', 
        34: 'Girar a la izquierda', 
        35: 'Solo adelante', 
        36: 'Ir recto o girar a la derecha', 
        37: 'Ir recto o girar a la izquierda', 
        38: 'Mantenerse a la derecha', 
        39: 'Mantenerse a la izquierda', 
        40: 'Circulación obligatoria en rotonda', 
        41: 'Fin de la prohibición de adelantar', 
        42: 'Fin de la prohibición de adelantar vehículos de más de 3.5 toneladas'
    }
    # Visualiza la matriz de confusión
    plt.figure(figsize=(15, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clases.values(), yticklabels=clases.values())
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()
'''


# Importación de bibliotecas
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
import os
from PIL import Image



# Función para cargar los datos
def load_data(train_csv, test_csv, image_folder):
    # Cargar los archivos CSV
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Obtener las rutas de las imágenes
    train_image_paths = train_data["Path"]
    test_image_paths = test_data["Path"]

    # Cargar las imágenes de entrenamiento
    X_train = []
    for image_path in train_image_paths:
        folder, subfolder, filename = image_path.split('/')
        image = Image.open(os.path.join(image_folder, folder, filename))
        image = image.resize((30, 30))
        X_train.append(np.array(image))

    # Cargar las imágenes de prueba
    X_test = []
    for image_path in test_image_paths:
        folder, subfolder, filename = image_path.split('/')
        image = Image.open(os.path.join(image_folder, folder, filename))
        image = image.resize((30, 30))
        X_test.append(np.array(image))

    # Convertir listas en arreglos de NumPy
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Obtener etiquetas
    y_train = to_categorical(train_data["ClassId"].values, 43)

    return X_train, y_train, X_test


# Función para crear el modelo CNN
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30, 30, 3)))
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
    return model

# Función para compilar el modelo
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Función para entrenar el modelo
def train_model(model, X_train, y_train, epochs=25, batch_size=32, validation_split=0.2):
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    return model

if __name__ == "__main__":
    train_csv = 'train.csv'  # Ruta al archivo CSV de entrenamiento
    test_csv = 'test.csv'    # Ruta al archivo CSV de prueba
    image_folder = r'C:\Users\pjime\Documents\ESTUDIOS\GRADO MATEMATICAS\4-CUARTO\TFG\GITHUB\CODIGOS\Reconocimiento de señales'   # Carpeta que contiene las imágenes

    X_train, y_train, X_test = load_data(train_csv, test_csv, image_folder)
    model = create_model()
    compile_model(model)

    # Entrenar el modelo
    train_model(model, X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)

    # Guardar el modelo
    model.save('traffic_classifier.h5')





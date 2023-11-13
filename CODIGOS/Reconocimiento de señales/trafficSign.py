'''
https://www.kaggle.com/code/yacharki/traffic-signs-image-classification-96-cnn#6.-Training-the-Model
Me he descargado los datos de esta pagina y he seguido mas o menos las instrucciones
'''


import numpy as np
import pandas as pd
import tensorflow as tf
import os
import cv2
from PIL import Image
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import time
import datetime

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
    model = build_model(X_train.shape[1:])
    train_model(model, 'traffic_classifier.keras', X_train, y_train, X_test, y_test)


'''
# Label Overview
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited'
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    """Carga los conjuntos de datos de entrenamiento y prueba desde archivos CSV."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess_data(train, test):
    """Realiza el preprocesamiento de los datos."""
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    X_train = X_train / 255.0
    X_train = X_train.values.reshape(-1, 28, 28, 1)

    Y_train = to_categorical(Y_train, num_classes=10)

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=2)

    return X_train, X_val, Y_train, Y_val

def define_model():
    """Define el modelo de red neuronal convolucional."""
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model

def compile_model(model):
    """Compila el modelo."""
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

def data_augmentation(X_train):
    """Realiza la técnica de aumento de datos."""
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=5,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )
    datagen.fit(X_train)
    return datagen

def train_model(model, datagen, X_train, Y_train, X_val, Y_val, batch_size, epochs):
    """Entrena el modelo y retorna el historial del entrenamiento."""
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                                  epochs=epochs, validation_data=(X_val, Y_val), steps_per_epoch=X_train.shape[0] // batch_size)
    return history

def plot_loss(history):
    """Visualiza las curvas de pérdida en el conjunto de validación."""
    plt.plot(history.history['val_loss'], color='b', label="validation loss")
    plt.title("Test Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, X_val, Y_val):
    """Visualiza la matriz de confusión."""
    Y_pred = model.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# Uso de las funciones
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train, test = load_data(train_path, test_path)
X_train, X_val, Y_train, Y_val = preprocess_data(train, test)

model = define_model()
compile_model(model)

datagen = data_augmentation(X_train)

batch_size = 250
epochs = 10

history = train_model(model, datagen, X_train, Y_train, X_val, Y_val, batch_size, epochs)

plot_loss(history)
plot_confusion_matrix(model, X_val, Y_val)

model.save('modelo.h5')

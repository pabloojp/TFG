import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')

data_dir = r'C:\Users\pjime\Documents\ESTUDIOS\GRADO MATEMATICAS\4-CUARTO\TFG\GITHUB\CODIGOS\Reconocimiento de señales'
train_path = r'C:\Users\pjime\Documents\ESTUDIOS\GRADO MATEMATICAS\4-CUARTO\TFG\GITHUB\CODIGOS\Reconocimiento de señales/Train'
test_path = r'C:\Users\pjime\Documents\ESTUDIOS\GRADO MATEMATICAS\4-CUARTO\TFG\GITHUB\CODIGOS\Reconocimiento de señales/Test'

# Resizing the images to 30x30x3
IMG_HEIGHT = 30
IMG_WIDTH = 30
channels = 3

NUM_CATEGORIES = len(os.listdir(train_path))
print(NUM_CATEGORIES)

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

folders = os.listdir(train_path)

train_number = []
class_num = []

for folder in folders:
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(clases[int(folder)])
    
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

# Visualizing 25 random images from test data
import random
from matplotlib.image import imread

test = pd.read_csv(data_dir + '/Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = data_dir + '/' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(b=None)
    plt.xlabel(rand_img.shape[1], fontsize = 20)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 20)#height of image



image_data = []
image_labels = []

# Recorremos cada clase
for i in range(NUM_CATEGORIES):
        path = os.path.join(data_dir, 'Train', str(i))  # Construimos la ruta para cada clase
        images = os.listdir(path)                             # Listamos todas las imágenes en la carpeta de la clase

        # Recorremos cada imagen en la clase
        for a in images:
                image = Image.open(os.path.join(path,  a))   # Cargamos la imagen utilizando PIL
                image_fromarray = Image.fromarray(image, 'RGB')
                resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))    # Redimensionamos la imagen a 30x30 píxeles
                image_data.append(np.array(resize_image))     # Convertimos la imagen a un array de NumPy y la agregamos la imagen a la lista de datos
                image_labels.append(i)           # Agregamos la etiqueta correspondiente a la lista de etiquetas


# Changing the list to numpy array
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]


X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)

X_train = X_train/255 
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)


y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)

print(y_train.shape)
print(y_val.shape)


model = keras.models.Sequential([    
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(43, activation='softmax')
])


lr = 0.001
epochs = 30

opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))


test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255


pred = model.predict_clases(X_test)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(labels, pred)*100)


from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred)
import seaborn as sns
df_cm = pd.DataFrame(cf, index = clases,  columns = clases)
plt.figure(figsize = (20,20))
sns.heatmap(df_cm, annot=True)


from sklearn.metrics import classification_report

print(classification_report(labels, pred))


plt.figure(figsize = (25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[start_index + i]
    actual = labels[start_index + i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    plt.imshow(X_test[start_index + i])
plt.show()
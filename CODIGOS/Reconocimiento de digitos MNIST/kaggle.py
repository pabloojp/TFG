"""
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
Alumno: Jiménez Poyatos, Pablo


Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento y de prueba como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.
"""


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
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical                     # Se importa la clase to_categorical, que realiza la codificación one-hot de las etiquetas en problemas de clasificación multiclase. La codificación one-hot es una técnica que convierte etiquetas categóricas en una representación numérica que es más adecuada para su uso en algoritmos de aprendizaje automático, especialmente en redes neuronales.
from keras.models import Sequential                                   # Se importa la clase Sequential de keras.models. Sequential es un tipo de modelo en el que las capas se apilan una encima de la otra en una secuencia lineal.
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D   # Se importan varias capas de redes neuronales desde el módulo layers de Keras. Estas capas son componentes fundamentales para construir modelos de redes neuronales convolucionales (CNN). Incluyen capas densas (Dense), capas de eliminación (Dropout), capas de aplanamiento (Flatten), capas convolucionales (Conv2D) y capas de agrupación máxima (MaxPool2D).
from keras.optimizers import RMSprop                                  # Se importa el optimizador RMSprop, que es un algoritmo de optimización utilizado para ajustar los pesos y los sesgos de una red neuronal durante el entrenamiento.
from keras.preprocessing.image import ImageDataGenerator              # Se importa ImageDataGenerator, que es una clase que se utiliza para realizar la generación de imágenes aumentadas y realizar la preprocesamiento de imágenes para su uso en el entrenamiento de modelos de redes neuronales. Esto es útil para aumentar la cantidad de datos de entrenamiento y mejorar la capacidad del modelo para generalizar.
from keras.callbacks import ReduceLROnPlateau                         # Se importa ReduceLROnPlateau, que es una devolución de llamada utilizada para reducir la tasa de aprendizaje durante el entrenamiento si ciertas condiciones no se cumplen.


# Configuramos el estilo Seaborn
sns.set(style='white', context='paper', palette='deep')     # Configura las preferencias de estilo de Seaborn


# Cargamos los datos de tipo CSV que he descargado en la web Kaggle
train = pd.read_csv("train.csv")   # 42000 ejemplos con 784 pixeles (imagenes 28x28). En train, las dimensiones son 42001 x 785
test = pd.read_csv("test.csv")     # 28000 ejemplos de test con 784 pixeles.


# Inicializamos las variables Y_train (etiquetas de cada ejemplo) y X_train (caracteristicas).
Y_train = train["label"]                           # Inicializamos la variable Y_train que contiene la primera columna de los datos introducidos en train.csv. En ella se almacena el numero correspondiente (etiquetas) a cada ejemplo de train.
X_train = train.drop(labels = ["label"],axis = 1)  # Caracteristicas de entrada de cada ejemplo sin las etiquetas de cada ejemplo.
del train                                          # Para eliminar espacio, como ya tenemos la información relevante en Y_train y x_train, eliminamos train.



# Muestra la distribución de las etiquetas de cada tipo en un grafico.
g = sns.countplot(Y_train)


# Calcula el recuento de cada clase
Y_train.value_counts()


# Comprobaciones de los datos y representación en una tabla de los datos vacios.
X_train.isnull().any().describe()


# Lo mismo pero de los tests
test.isnull().any().describe()


# Normalizamos la escala de grises para reducir la iluminación. Además, las CNN convergen más rápidamente con datos en el rango [0..1] que en el rango [0..255].
X_train = X_train / 255.0
test = test / 255.0


# Las imágenes de entrenamiento y prueba, que tienen un tamaño de 28 píxeles x 28 píxeles, se han almacenado en un DataFrame de pandas como vectores unidimensionales de 784 valores. Este paso, nos devuelve los datos como matrices 28x28x1 (1 canal, escala grises)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# Codificamos las etiquetas en vectores "one-hot". Cada etiqueta la representamos como un vector binario en el que un solo elemento tiene un valor de 1 y los demás son 0.
Y_train = to_categorical(Y_train, num_classes = 10)


# Establecemos la semilla aleatoria en 2. Dividimos el conjunto de entrenamiento en dos partes: la validación (10%) y el entreno (90%), para asegurar que el modelo pueda ser evaluado en datos que no ha visto durante el entrenamiento. La división aleatoria garantiza que no haya un desequilibrio en las etiquetas entre el conjunto de entrenamiento y el conjunto de validación. 
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)


# Algunos ejemplos
g = plt.imshow(X_train[0][:,:,0])




# Creaos el modelo de CNN. La arquitectura es : In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))   # SAME : la imagen de salida tendrá las mismas dimensiones espaciales que la imagen de entrada.
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))
# Filtros de 32 canales: Los filtros son pequeñas matrices que se utilizan para escanear partes de la imagen de entrada. En este caso, se están utilizando 32 filtros en ambas capas. Cada filtro es como una "lente" que busca patrones específicos en la imagen.
# Kernel de convolución 5x5: El kernel de convolución es una matriz que se desliza sobre la imagen para realizar una operación matemática llamada convolución. En este caso, se está utilizando un kernel de 5x5, lo que significa que cada filtro analiza una región de 5x5 píxeles de la imagen en cada paso.
# Activación 'relu': 'relu' es una función de activación llamada Rectified Linear Unit. Después de que cada filtro haya realizado la convolución con la región de la imagen, se aplica la función 'relu'. Esta función es muy simple: si el valor de salida de la convolución es positivo, se mantiene tal cual; si es negativo, se establece en cero. Esto introduce no linealidad en la red y permite que la CNN aprenda características más complejas y patrones en los datos.
# Entonces, n estas dos capas, cada uno de los 32 filtros busca patrones en regiones de 5x5 píxeles de la imagen de entrada y aplica la función 'relu' para resaltar las características relevantes. Estas operaciones se realizan para cada filtro, lo que permite que la CNN extraiga múltiples características de la imagen.
model.add(MaxPool2D(pool_size=(2,2)))  # Capa MaxPool2D: La capa MaxPool2D es una capa de submuestreo que reduce la dimensionalidad de las características extraídas por las capas de convolución. En esta configuración, se utiliza una ventana de 2x2 píxeles (pool_size=(2,2)). Lo que hace esta capa es examinar grupos de 2x2 píxeles en las características de la capa anterior y retiene solo el valor máximo de esos 4 píxeles. Este proceso reduce la cantidad de información y cálculos en la red, lo que ayuda a evitar el sobreajuste y mejora el rendimiento computacional.
model.add(Dropout(0.25))   # Capa Dropout: La capa Dropout es una técnica de regularización. La regularización es una forma de prevenir el sobreajuste en la red. En esta capa, se establece una fracción de las unidades de la capa anterior en cero de manera aleatoria durante el entrenamiento. En este caso, el valor es 0.25, lo que significa que aproximadamente el 25% de las unidades de esta capa se establecerán en cero en cada paso de entrenamiento. Esto fuerza a la red a aprender de manera más robusta y generalizable, ya que no puede depender demasiado de ninguna unidad específica.

# Repetimos el mismo proceso pero ahora con algunas modificaciones.
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))                 
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())                           #  Esta capa toma las características generadas por las capas de convolución y las convierte en un vector unidimensional.
model.add(Dense(256, activation = "relu"))     # Se agrega una capa con 256 neuronas y activacion relu. Esta capa es una capa completamente conectada y se utiliza para aprender patrones más globales de las características extraídas por las capas de convolución.
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))   # Esta es la capa de salida que produce la distribución de probabilidad de las 10 clases posibles.



# Una vez que nuestras capas se han agregado al modelo, necesitamos configurar una función de puntaje, una función de pérdida y un algoritmo de optimización.


# Defino el optimizador
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, centered=False)


# Compilamos el modelo
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# Definimos un programador de tasa de aprendizaje
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)


epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86



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


# Entrenamos nustro modelo
datagen.fit(X_train)
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size), epochs = epochs, validation_data = (X_val,Y_val), verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size , callbacks=[learning_rate_reduction])
                            

model.save('modelo_digitos1.keras')

'''
# Visualización de la matriz de confusión. Muestra la matriz de confusión con opciones para normalizar y personalizar el título y el mapa de colores.
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de confusión', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Etiqueta real')
    plt.xlabel('Etiqueta predicha')


# Predecir los valores del conjunto de validación
Y_pred = model.predict(X_val)


# Convertir las clases de predicción a vectores one-hot
Y_pred_classes = np.argmax(Y_pred,axis = 1) 


# Convertir observaciones de validación a vectores one-hot
Y_true = np.argmax(Y_val,axis = 1) 


# Calcular la matriz de confusión
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 


# Mostrar la matriz de confusión
plot_confusion_matrix(confusion_mtx, classes = range(10)) 



# Mostrar algunos resultados de error

# Los errores son las diferencias entre las etiquetas predichas y las etiquetas reales
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """Esta función muestra 6 imágenes con sus etiquetas predichas y reales"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Etiqueta predicha: {}\nEtiqueta real: {}".format(pred_errors[error], obs_errors[error]))
            n += 1


# Probabilidades de los números predichos incorrectamente
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)


# Probabilidades predichas de los valores reales en el conjunto de errores
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))


# Diferencia entre la probabilidad de la etiqueta predicha y la etiqueta real
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors


# Lista ordenada de los errores de probabilidad delta
sorted_dela_errors = np.argsort(delta_pred_true_errors)


# 6 errores más importantes
most_important_errors = sorted_dela_errors[-6:]


# Mostrar los 6 errores más importantes
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)


# Predecir resultados
results = model.predict(test)


# Seleccionar el índice con la máxima probabilidad
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")


# submission combina el número de imagen y las etiquetas predichas en un archivo CSV para su sumisión.
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)

'''
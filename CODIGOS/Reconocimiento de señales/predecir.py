"""
Nombre del codigo: Modelo CNN reconocimiento de dígitos usando dataset MNIST.
Guiado por: Tutorial de Kaggle (acceso al enlace el 10 de noviembre)
        https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6/notebook
Alumno: Jiménez Poyatos, Pablo

Script solo con el modelo. Nada de representación de datos ni nada. Además el codigo apilado en funciones.

Para crear el modelo, he necesitado instalarme diferentes bibliotecas como numpy, tensorflow, keras, etc.

Además, he tenido que descargarme los datos de entrenamiento y de prueba como archivos CSV y guardarlos en
la misma carpeta donde estaba este script.

Para predecir el digito que aparece en una imagen, tenemos que cargar la imagen cuyo tamaño sea 28 x 28.
Si no esta en escala de grises, la convertimos.
"""

from PIL import Image
import numpy as np
from keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('traffic_classifier.keras')

# Cargar la imagen '41.png'
image_path = '41.png'
image = Image.open(image_path)
image = image.resize((30, 30))
image = image.convert('RGB')
image = np.array(image)
image = image / 255.0  # Normalizar los valores de píxeles

# Realizar la predicción con el modelo cargado
prediction = model.predict(np.array([image]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)

# Obtener la etiqueta predicha (índice de la clase con mayor probabilidad)
predicted_class = np.argmax(prediction)

# Crear un diccionario que mapea las clases a sus etiquetas
classes = { 
            0:'Speed limit (20km/h)',
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
            16:'Veh > 3.5 tons prohibited',
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
            42:'End no passing veh > 3.5 tons'
        }

# Obtener la etiqueta correspondiente a la clase predicha
predicted_label = classes[predicted_class]

# Imprimir la etiqueta predicha
print(f'Clase predicha: {predicted_label}')



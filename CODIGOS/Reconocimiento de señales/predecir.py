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

# Imprimir la etiqueta predicha
print(f'Clase predicha: {predicted_class}')



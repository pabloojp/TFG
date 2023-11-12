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

from keras.models import load_model
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = load_model('modelo_digitos.keras')

# Cargar la imagen que deseas predecir
from PIL import Image

# Cargar la imagen y convertirla a escala de grises si es necesario y con tamaño 28x28
image = Image.open('numerocero.png').convert('L')
image = image.resize((28, 28))

# Preprocesar la imagen
image = np.array(image) / 255.0
image = image.reshape(-1, 28, 28, 1)

# Realizar la predicción
prediction = loaded_model.predict(image)

# Obtener el número predicho
predicted_number = np.argmax(prediction)

print(f'El número predicho es: {predicted_number}')
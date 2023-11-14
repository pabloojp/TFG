"""
Nombre del codigo: Prediccion digito usando el modelo creado. 
Alumno: Jiménez Poyatos, Pablo

Para predecir el digito que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con un solo canal, la escala de grises y redimensionarla 
a 28x28 pixeles.
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
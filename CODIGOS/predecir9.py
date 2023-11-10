from keras.models import load_model
import numpy as np

# Cargar el modelo desde el archivo
loaded_model = load_model('modelo_digitos.h5')

# Cargar la imagen que deseas predecir
from PIL import Image

# Cargar la imagen y convertirla a escala de grises si es necesario
image = Image.open('numeronueve.png').convert('L')

# Preprocesar la imagen
image = np.array(image) / 255.0
image = image.reshape(-1, 28, 28, 1)

# Realizar la predicción
prediction = loaded_model.predict(image)

# Obtener el número predicho
predicted_number = np.argmax(prediction)

print(f'El número predicho es: {predicted_number}')
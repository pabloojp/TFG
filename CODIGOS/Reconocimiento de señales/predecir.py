"""
Nombre del codigo: Prediccion señal de trafico usando el modelo creado. 
Alumno: Jiménez Poyatos, Pablo

Para predecir la señal que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con tres canales  y redimensionarla a 30 x 30 pixeles.
Despues de que mi modelo prediga cual sería su etiqueta, le asignamos su nombre correspondiente del diccionario 
clases.
"""

from PIL import Image
import numpy as np
from keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('traffic_classifier.keras')

# Cargar la imagen
image_path = 'prohib.jpg'
image = Image.open(image_path)
image = image.resize((30, 30))
image = image.convert('RGB')
image = np.array(image)
  # Normalizar los valores de píxeles

# Realizar la predicción con el modelo cargado
prediction = model.predict(np.array([image]))  # Asegurarse de que sea un arreglo de forma (1, 30, 30, 3)

# Obtener la etiqueta predicha (índice de la clase con mayor probabilidad)
predicted_class = np.argmax(prediction)

# Crear un diccionario que mapea las clases a sus etiquetas
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

# Obtener la etiqueta correspondiente a la clase predicha
predicted_label = clases[predicted_class]

probabilities = prediction[0]

# Imprimir la etiqueta predicha
print(f'Clase predicha: {predicted_label}')

print('Probabilidades:')
for i, prob in enumerate(probabilities):
    print(f'{clases[i]}: {prob:.4f}')



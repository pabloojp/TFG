from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

def convertir_a_escala_de_grises(imagen_path):
    imagen = Image.open(imagen_path)
    imagen = imagen.convert('L')  # Convertir a escala de grises

    return imagen

def invertir_colores(imagen):
    # Invertir los colores
    imagen = Image.eval(imagen, lambda x: 255 - x)

    return imagen

def tiene_fondo_negro(imagen, umbral=100):
    # Convertir la imagen a un array de numpy
    imagen_np = np.array(imagen)

    # Calcular el promedio de intensidad de píxeles
    promedio_intensidad = np.mean(imagen_np)

    # Comparar el promedio con el umbral
    if promedio_intensidad < umbral:
        return True
    else:
        return False

if __name__ == "__main__":
    # Cargar el modelo desde el archivo
    loaded_model = load_model('modelo_digitos.keras')

    ruta_imagen_inicial = 'numverd3.jpg'

    # Convertir la imagen a escala de grises
    imagen_gris = convertir_a_escala_de_grises(ruta_imagen_inicial)
    imagen_gris.show()

    # Verificar si la imagen tiene fondo negro
    es_fondo_negro = tiene_fondo_negro(imagen_gris)

    print(es_fondo_negro)

    # Si tiene fondo negro, invertir los colores
    if not es_fondo_negro:
        imagen_gris = invertir_colores(imagen_gris)

    # Guardar la imagen invertida si es necesario
    if not es_fondo_negro:
        nueva_ruta = "imagen_invertida.jpg"
        imagen_gris.save(nueva_ruta)
    else:
        nueva_ruta = ruta_imagen_inicial

    # Cargar la imagen con tamaño 28x28
    image = imagen_gris.resize((28, 28))

    # Preprocesar la imagen
    image = np.array(image) / 255.0
    image = image.reshape(-1, 28, 28, 1)

    # Realizar la predicción
    prediction = loaded_model.predict(image)

    # Obtener el número predicho
    predicted_number = np.argmax(prediction)

    print(f'El número predicho es: {predicted_number}')

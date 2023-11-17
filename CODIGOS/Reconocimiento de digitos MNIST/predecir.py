"""
Nombre del codigo: Prediccion digito usando el modelo creado. 
Alumno: Jiménez Poyatos, Pablo

Para predecir el digito que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con un solo canal, la escala de grises y redimensionarla 
a 28x28 pixeles.
"""

from keras.models import load_model
import numpy as np
from PIL import Image
import cv2


def invertir_colores(imagen_path):
    imagen = Image.open(imagen_path)
    imagen = imagen.convert('RGB')
    
    # Invertir los colores
    imagen = Image.eval(imagen, lambda x: 255 - x)

    # Guardar la nueva imagen
    nueva_ruta = "imagen_invertida.jpg"
    imagen.save(nueva_ruta)

    return nueva_ruta

def tiene_fondo_negro(imagen_path, umbral=100):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    # Calcular el promedio de intensidad de píxeles
    promedio_intensidad = np.mean(imagen)

    # Comparar el promedio con el umbral
    if promedio_intensidad < umbral:
        return True
    else:
        return False





if __name__ == "__main__":

    # Cargar el modelo desde el archivo
    loaded_model = load_model('modelo_digitos.keras')

    ruta_imagen_inicial = 'numverd2.jpg'

    es_fondo_negro = tiene_fondo_negro(ruta_imagen_inicial)

    print (es_fondo_negro)

    ruta_imagen = invertir_colores(ruta_imagen_inicial)

    if es_fondo_negro:
        ruta_imagen = ruta_imagen_inicial
    
    # Cargar la imagen y convertirla a escala de grises si es necesario y con tamaño 28x28
    image = Image.open(ruta_imagen).convert('L')
    image.show()
    image = image.resize((28, 28))

    # Preprocesar la imagen
    image = np.array(image) / 255.0
    image = image.reshape(-1, 28, 28, 1)

    # Realizar la predicción
    prediction = loaded_model.predict(image)

    # Obtener el número predicho
    predicted_number = np.argmax(prediction)

    print(f'El número predicho es: {predicted_number}')

'''
    """
Nombre del codigo: Prediccion digito usando el modelo creado. 
Alumno: Jiménez Poyatos, Pablo

Para predecir el digito que aparece en una imagen, tenemos que cargar el modelo despues de 20 epochs y
la imagen. Esta última tenemos que pasarla a una imagen con un solo canal, la escala de grises y redimensionarla 
a 28x28 pixeles.
"""

from keras.models import load_model
import numpy as np
from PIL import Image
import cv2


def invertir_colores(imagen_path):
    imagen = Image.open(imagen_path)
    imagen = imagen.convert('RGB')
    
    # Invertir los colores
    imagen = Image.eval(imagen, lambda x: 255 - x)

    # Guardar la nueva imagen
    nueva_ruta = "imagen_invertida.jpg"
    imagen.save(nueva_ruta)

    return nueva_ruta

def tiene_fondo_negro(imagen_path, umbral=100):
    # Cargar la imagen en escala de grises
    imagen = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)

    # Calcular el promedio de intensidad de píxeles
    promedio_intensidad = np.mean(imagen)

    # Comparar el promedio con el umbral
    if promedio_intensidad < umbral:
        return True
    else:
        return False





if __name__ == "__main__":

    # Cargar el modelo desde el archivo
    loaded_model = load_model('modelo_digitos.keras')

    ruta_imagen_inicial = 'numb0.png'

    imagen= Image.open(ruta_imagen_inicial)
    imagen.show()
    image = imagen.convert('L')
    image.show()
    nueva_ruta = "imagen_bn.jpg"
    image.save(nueva_ruta)

    es_fondo_negro = tiene_fondo_negro(nueva_ruta)

    print (es_fondo_negro)

    ruta_imagen = ruta_imagen_inicial

    if not es_fondo_negro:
        ruta_imagen = invertir_colores(nueva_ruta)
        
    imagen_actualizada= Image.open(ruta_imagen)
    # Cargar la imagen y convertirla a escala de grises si es necesario y con tamaño 28x28

    imagen_actualizada = imagen_actualizada.resize((28, 28))
    imagen_actualizada.show()

    # Preprocesar la imagen
    imagen_actualizada = np.array(imagen_actualizada) / 255.0
    imagen_actualizada = imagen_actualizada.reshape(-1, 28, 28, 1)

    # Realizar la predicción
    prediction = loaded_model.predict(imagen_actualizada)

    print(prediction)

    # Obtener el número predicho
    predicted_number = np.argmax(prediction)

    print(f'El número predicho es: {predicted_number}')
'''

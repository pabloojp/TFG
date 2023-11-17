"""
Asignatura: Informática 
Curso: 2020-2021
Ejercicios: Práctica 3. Función convert_to_L().
"""

'''
# Ruta de la imagen que quieres convertir
ruta_imagen = 'ruta/de/tu/imagen.jpg'

# Abre la imagen utilizando PIL
imagen_original = Image.open(ruta_imagen)

# Llama a tu función convert_to_L
imagen_convertida = convert_to_L(imagen_original)

# Guarda la imagen convertida si es necesario
imagen_convertida.save('ruta/de/imagen/convertida.jpg')

# Puedes mostrar la imagen convertida si lo deseas
imagen_convertida.show()
'''

from PIL import Image

def convert_to_L(ruta_imagen):
    imagen = Image.open(ruta_imagen)
    imagenResultado = Image.new('L', imagen.size)
    width, height = imagen.size

    minimo, maximo = imagen.getextrema()

    intervalo = float(maximo - minimo)

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            pixel = imagen.getpixel((x, y))
            imagenResultado.putpixel((x, y), round(((pixel - minimo) / intervalo) * 255))

    return imagenResultado

ruta_imagen = r'C:\Users\pjime\Documents\ESTUDIOS\GRADO MATEMATICAS\4-CUARTO\TFG\GITHUB\CODIGOS\Reconocimiento de digitos MNIST/numero6.png'
imagen_convertida = convert_to_L(ruta_imagen)



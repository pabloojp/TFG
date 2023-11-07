import numpy as np
from scipy import stats

class capa():
  def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
    self.funcion_act = funcion_act
    self.b  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas).reshape(1,n_neuronas),3)                                     #Se generan valores aleatorios para los bias ("b") de la capa. Estos valores se obtienen a partir de una distribución truncada normal en el rango [-1, 1]. Los valores se redondean a tres decimales y se almacenan en b.
    self.W  = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior,n_neuronas),3)   #Se generan valores aleatorios para los pesos ("W") de la capa. Al igual que los bias, se obtienen a partir de una distribución truncada normal en el rango [-1, 1]. Los valores se redondean a tres decimales y se almacenan en W.

import math
import matplotlib.pyplot as plt


sigmoid = (                           # Creo una tupla que contiene dos funciones.
  lambda x:1 / (1 + np.exp(-x)),      # Calcula la funcion de activacion sigmoid para un valor de entrada x
  lambda x:x * (1 - x)                # Calcula la derivada de la funcion sigmoid
  )

rango = np.linspace(-10,10).reshape([50,1])   # Se genera un rango de valores lineales que van desde -10 hasta 10 y se almacenan en la variable "rango". Luego, se le da forma a un arreglo bidimensional de 50 filas y 1 columna.


datos_sigmoide = sigmoid[0](rango)            # Se calculan los valores de la función de activación sigmoid en el rango generado y se almacenan en la variable datos_sigmoide.
datos_sigmoide_derivada = sigmoid[1](rango)   # Se calculan los valores de la derivada de la función de activación sigmoid en el mismo rango y se almacenan en "datos_sigmoide_derivada".


'''
Creamos los graficos

fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
fig.tight_layout()
'''


def derivada_relu(x):
  x[x<=0] = 0
  x[x>0] = 1
  return x



relu = (
  lambda x: x * (x > 0),                # Anula los valores negativos y los positivos los mantiene igual.
  lambda x: derivada_relu(x)
)



datos_relu = relu[0](rango)           # Se calculan los valores de la función de activación ReLU en el rango previamente definido y se almacenan en la variable "datos_relu".
datos_relu_derivada = relu[1](rango)  # Se calculan los valores de la derivada de la función de activación ReLU en el mismo rango y se almacenan en la variable "datos_relu_derivada".


# Volvemos a definir rango que ha sido cambiado
rango = np.linspace(-10,10).reshape([50,1])     


'''
Creamos los graficos
plt.cla()                                                     # Se limpia el gráfico actual, si es que hubiera alguno, para preparar la creación de nuevos gráficos.               
fig, axes = plt.subplots(nrows=1, ncols=2, figsize =(15,5))   # Se crea una nueva figura con dos subgráficos. La variable fig representa la figura y axes contiene los subgráficos. 

axes[0].plot(rango, datos_relu[:,0])                          # Se traza la función de activación ReLU en el primer subgráfico ("axes[0]") para la primera columna de "datos_relu". 
axes[1].plot(rango, datos_relu_derivada[:,0])                 # Se traza la derivada de la función de activación ReLU en el segundo subgráfico ("axes[1") para la primera columna de "datos_relu_derivada".
plt.show()                                                    # Se muestra la figura con los dos gráficos. La función show() permite visualizar los gráficos en la pantalla.
'''


# Numero de neuronas en cada capa. 
# El primer valor es el numero de columnas de la capa de entrada.
neuronas = [2,4,8,1]   # Capa de entrada, capas ocultas y capa final.

# Funciones de activacion usadas en cada capa. 
funciones_activacion = [relu,relu, sigmoid]


red_neuronal = []

for paso in range(len(neuronas)-1):
  x = capa(neuronas[paso],neuronas[paso+1],funciones_activacion[paso])   # Crea una instancia de la clase "capa
  red_neuronal.append(x)

print(red_neuronal)




# Ejemplo de vector de entrada

X =  np.round(np.random.randn(20,2),3)   # Matriz de datos de entrada 20x2
 
z = X @ red_neuronal[0].W                # @ multiplica matrices. Se calcula z multiplicando la matriz de entrada X por los pesos ("W") de la primera capa de la red neuronal ("red_neuronal[0]"). Esto representa la suma ponderada de los valores de entrada.

# print(z[:10,:], X.shape, z.shape)


z = z + red_neuronal[0].b                # Se añaden los bias ("b") de la primera capa a "z". Esto representa la suma ponderada de los valores de entrada con los sesgos.

# print(z[:5,:])

a = red_neuronal[0].funcion_act[0](z)             # Se aplica la función de activación de la primera capa (ReLU en este caso) a "z" y se almacena en "a". 
# a[:5,:]

output = [X]

for num_capa in range(len(red_neuronal)):
  z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b        # Calculo "z" aplicando la matriz de pesos ("W") y sumando los sesgos ("b") a la salida de la capa anterior, que se toma de "output[-1]".
  a = red_neuronal[num_capa].funcion_act[0](z)                                # Aplico la función de activación de la capa a "z" y se almacena en "a".
  output.append(a)                                                            # Agrego "a" a la lista output.

print(output[-1])


def mse(Ypredich, Yreal):

  # Calculamos el error cuadratico medio.
  x = (np.array(Ypredich) - np.array(Yreal)) ** 2    # np.array(Ypredich) y np.array(Yreal) convierten las predicciones y los valores reales en arreglos NumPy si no lo son.
  x = np.mean(x)

  # Error simple
  y = np.array(Ypredich) - np.array(Yreal)
  return (x,y)


from random import shuffle

Y = [0] * 10 + [1] * 10
shuffle(Y)                # Barajan de forma aleatoria.
Y = np.array(Y).reshape(len(Y),1)

print(mse(output[-1], Y)[0])



# El resultado final es la salida de la red neuronal en el último paso, que es una matriz de 20 filas y 1 columna con valores que representan las predicciones de la red para cada ejemplo de entrada. El valor impreso en la última línea del código es el MSE entre estas predicciones y las etiquetas reales Y.



'''
Esta red neuronal es un ejemplo simple de una red feedforward, que toma valores 
de entrada, realiza una serie de operaciones en capas y produce una salida. 
Vamos a desglosar paso a paso lo que hace esta red neuronal:

Definición de la Red Neuronal:
Se define la estructura de la red neuronal especificando el número de neuronas 
en cada capa, así como las funciones de activación utilizadas en cada capa. 
En este caso, la red tiene 3 capas: una capa de entrada con 2 neuronas, una 
capa oculta con 4 neuronas, otra capa oculta con 8 neuronas y una capa de 
salida con 1 neurona. Las funciones de activación son ReLU (Rectified Linear 
Unit) para las capas ocultas y la función sigmoid para la capa de salida.

Creación de Capas:
Se crean instancias de la clase capa para cada capa de la red neuronal. Cada 
capa tiene pesos (W) y sesgos (b) asociados, que se inicializan con valores aleatorios.

Generación de Datos de Entrada:
Se genera una matriz de datos de entrada X con dimensiones 20x2 para este 
ejemplo. Esto simula los datos que se alimentarán a la red.

Propagación hacia Adelante:
Se realiza la propagación hacia adelante (forward propagation) a través de la 
red. Esto implica:
-Multiplicar los datos de entrada X por los pesos (W) de la primera capa.
-Sumar los sesgos (b) de la primera capa.
-Aplicar la función de activación ReLU a la salida de la primera capa.
-Repetir estos pasos para cada capa de la red hasta llegar a la capa de salida.

Cálculo del Error:
Se compara la salida final de la red con los valores reales (en este caso, 
etiquetas de clase binarias) para calcular el error cuadrático medio (MSE).

Retropropagación del Error (no se muestra en el código):
En un entrenamiento real de una red neuronal, normalmente se utilizaría un 
algoritmo de retropropagación (backpropagation) para ajustar los pesos y sesgos 
de la red y minimizar el error.

En resumen, esta red neuronal toma un conjunto de datos de entrada, realiza 
cálculos a través de múltiples capas utilizando pesos y sesgos, aplica 
funciones de activación en cada capa y produce una salida final. La salida 
puede ser utilizada para hacer predicciones en problemas de clasificación o 
regresión, y el entrenamiento se realizaría para ajustar los pesos y sesgos de 
la red para que se ajusten mejor a los datos de entrenamiento. El código 
proporcionado se centra en la definición y propagación de la red, pero falta la 
parte de entrenamiento y retropropagación para hacer que la red aprenda a 
partir de los datos.
'''


'''
Resumen del código:
    
    Este ejemplo es una implementación simple de una red neuronal en Python 
    usando feedforward. El objetivo es comprender los conceptos básicos de las 
    redes neuronales y cómo funcionan.

    1. En el código, primero se define una clase llamada "capa" que representa una 
    capa de la red neuronal. Cada capa tiene pesos (W) y sesgos (b) que se 
    inicializan aleatoriamente a partir de una distribución normal truncada. 
    También se permite especificar una función de activación para la capa.

    2. A continuación, se definen dos funciones de activación comunes: la función 
    sigmoide y la función ReLU. Se generan datos de ejemplo para estas 
    funciones y se crean gráficos para visualizarlas.

    3. Luego, se establece la arquitectura de la red neuronal, especificando el 
    número de neuronas en cada capa y las funciones de activación a utilizar en 
    cada capa. En este ejemplo, hay una capa de entrada con 2 neuronas, dos 
    capas ocultas con 4 y 8 neuronas, respectivamente, y una capa de salida con 
    1 neurona que utiliza la función sigmoide.
    
    4. Se crea una instancia de la clase "capa" para cada capa de la red y se 
    almacenan en una lista llamada "red_neuronal". Esto representa la 
    arquitectura de la red.
    
    5. Se generan datos de entrada de ejemplo en forma de una matriz 2D.
    
    6. Se realiza una propagación hacia adelante a través de la red neuronal, 
    calculando la salida de cada capa y aplicando las funciones de activación 
    correspondientes. La salida final se almacena en "output[-1]".
    
    7. Finalmente, se calcula el Error Cuadrático Medio (MSE) entre las 
    predicciones de la red y las etiquetas de clase de ejemplo. Se utiliza la 
    función "shuffle" para barajar aleatoriamente las etiquetas de clase.
    
    En resumen, este ejemplo ilustra la construcción y operación de una red 
    neuronal simple utilizando Python y NumPy. Es un punto de partida para 
    comprender cómo funcionan las redes neuronales antes de profundizar en 
    conceptos más avanzados.
    
'''

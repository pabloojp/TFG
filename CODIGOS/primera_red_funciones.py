import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class Capa:
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas).reshape(1, n_neuronas), 3)
        self.W = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior, n_neuronas), 3)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def relu_derivative(x):
    return relu(x)

def create_activation_functions():
    sigmoid_activation = (sigmoid, sigmoid_derivative)
    relu_activation = (relu, relu_derivative)
    return sigmoid_activation, relu_activation

def plot_activation_functions(rango, activation_func, title):
    plt.plot(rango, activation_func(rango))
    plt.title(title)
    plt.show()

def create_neural_network(neuronas, funciones_activacion):
    red_neuronal = []
    for paso in range(len(neuronas) - 1):
        capa_actual = Capa(neuronas[paso], neuronas[paso + 1], funciones_activacion[paso])
        red_neuronal.append(capa_actual)
    return red_neuronal

def forward_propagation(X, red_neuronal):
    output = [X]
    for capa in red_neuronal:
        z = output[-1] @ capa.W + capa.b
        a = capa.funcion_act[0](z)
        output.append(a)
    return output[-1]

def mse(Ypredich, Yreal):
    squared_error = (np.array(Ypredich) - np.array(Yreal)) ** 2
    mse_value = np.mean(squared_error)
    error = np.array(Ypredich) - np.array(Yreal)
    return mse_value, error

def main():
    rango = np.linspace(-10, 10).reshape([50, 1])
    sigmoid_activation, relu_activation = create_activation_functions()

    # Plot the activation functions
    plot_activation_functions(rango, sigmoid_activation[0], "Sigmoid Activation")
    plot_activation_functions(rango, relu_activation[0], "ReLU Activation")

    # Number of neurons in each layer
    neuronas = [2, 4, 8, 1]
    funciones_activacion = [relu_activation, relu_activation, sigmoid_activation]

    red_neuronal = create_neural_network(neuronas, funciones_activacion)
    print(red_neuronal)

    X = np.round(np.random.randn(20, 2), 3)

    z = X @ red_neuronal[0].W
    z = z + red_neuronal[0].b
    a = red_neuronal[0].funcion_act[0](z)

    output = forward_propagation(X, red_neuronal)
    Y = [0] * 10 + [1] * 10
    np.random.shuffle(Y)
    Y = np.array(Y).reshape(len(Y), 1)

    mse_value, error = mse(output, Y)
    print("Mean Squared Error:", mse_value)
    print("Error:", error)

if __name__ == "__main__":
    main()


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

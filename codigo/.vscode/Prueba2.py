import math
import numpy as np
import matplotlib.pyplot as plt

T = np.zeros((3, 2))  # Creación de matrizes de transformación
E = np.zeros((3, 3))
j1 = np.zeros((2,3))
j2neg = np.zeros((2,2))

Aiz = math.pi/2 #Declaracion de variables
Biz = math.pi
Ade = -math.pi/2
Bde = 0
L = 80      #medidas en mm
Riz = 35
Rde = 35

velocidades_entrada = np.zeros(2)  # Creación de vector para velocidad lineal y angular
posicion_orientacion = np.zeros(3)  # Creación de vector para posición (X, Y) y orientación (Theta)

def cargar_archivo(nombre):
    try:
        with open(nombre, 'r') as file:
            valores = []
            for dato in file:
                valores.append([float(x) for x in dato.split(",")])

            velocidades_entrada[0] = valores[0][2]  # Asignación de valores respectivos para cada caso
            velocidades_entrada[1] = valores[0][3]
            posicion_orientacion[2] = valores[0][1]

            return valores[0][0]  # Devolver el tiempo de muestreo

    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo {nombre}.")
        return None
    except Exception as e:
        print(f"Error al cargar los parámetros desde el archivo: {e}")
        return None

def actualizar_posicion_orientacion(dt):
    global T, posicion_orientacion, velocidades_entrada, jacobiano2
    # Actualizar la matriz de transformación con la orientación actual
    T = np.array([[np.cos(posicion_orientacion[2]), np.sin(posicion_orientacion[2])],
                  [-np.sin(posicion_orientacion[2]), np.cos(posicion_orientacion[2])],
                  [0, dt]])

    # Creacion de la matriz de transformacion local
    E = np.array([[np.cos(posicion_orientacion[2]), -np.sin(posicion_orientacion[2]), 0],
                  [-np.sin(posicion_orientacion[2]), np.cos(posicion_orientacion[2]), 0],
                  [0, 0, dt]])
    
    # Creacion de matriz Jacobiana para la descripcion del movimiento de las ruedas
    j1 = np.array([[np.sin(Aiz + Biz), -np.cos(Aiz + Biz), -1*np.cos(Biz)],
                   [np.sin(Ade + Bde), -np.cos(Ade + Bde), -1*np.cos(Bde)]])
    
    j2neg = np.array([[1/Riz, 0],
                      [0, 1/Rde]])
    
    # Aplicar la matriz de transformación, movimiento en marco global
    resultado = np.matmul(T, velocidades_entrada)

    # Aplicar transformacion a movimiento en marco local
    var_local = np.matmul(E, resultado)

    #Aplicar matriz Jacobiana
    jacobiano = np.matmul(j1, var_local)
    jacobiano2 = np.matmul(jacobiano, j2neg)

    return jacobiano2

def guardar_resultado(dt, jacobiano2): #funcion para guardar el resultado
    try:
        datos = np.array([[dt] + jacobiano2.flatten().tolist()])

        # Guardar los datos en el archivo resultado.txt
        np.savetxt("resultado.txt", datos, fmt="%.4f", delimiter=",")
    except Exception as e:
        print(f"Error al guardar los datos en resultado.txt: {e}")

def main():
    global jacobiano2, dt
    dt = cargar_archivo("datos.txt")  # Leer parámetros desde el archivo datos.txt y obtener el tiempo de muestreo
    if dt is not None:
        print(f"Parámetros cargados correctamente. Tiempo: {dt} segundos.")
        print(f"Velocidad lineal (V): {velocidades_entrada[0]}")  # Mostrar parámetros leídos
        print(f"Velocidad angular (w): {velocidades_entrada[1]}")
    else:
        return

    tiempo_final = dt  # Tiempo total de simulación
    tiempo_muestras = np.arange(0, tiempo_final, 0.1)

    jacobiano2_samples = []

    for t in tiempo_muestras:
        # Actualizar posición y orientación en cada paso de tiempo
        jacobiano2 = actualizar_posicion_orientacion(t)
        guardar_resultado(t, jacobiano2)
        jacobiano2_samples.append(jacobiano2.flatten())

    jacobiano2_samples = np.array(jacobiano2_samples)

    # Descomponer jacobiano2 en sus componentes
    j1_11, j1_12 = jacobiano2_samples[:, 0], jacobiano2_samples[:, 1]

    # Gráficos
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(tiempo_muestras, j1_11, color='blue', label='llanta Der')
    axs[0].set_title('Gráfico de la velocidad llanta Derecha)')
    axs[0].legend()

    axs[0].set_xlabel('Tiempo(s)')

    axs[1].plot(tiempo_muestras, j1_12, color='red', label='llanta Izq')
    axs[1].set_title('Gráfico de la velocidad llanta Izquierda)')
    axs[1].legend()

    axs[1].set_xlabel('Tiempo(s)')

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
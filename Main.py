import numpy as np
from skimage.io import imshow, imread
import cv2
import os
import matplotlib.pyplot as plt

def recortar_imagen_v2(ruta_img: str, ruta_img_crop: str, x_inicial: int, x_final: int, y_inicial: int, y_final: int)-> None:
    """
    Esta función recibe una imagen y devuelve otra imagen recortada.

    Args:
      ruta_img (str): Ruta de la imagen original que se desea recortar.
      ruta_img_crop (str): Ruta donde se guardará la imagen recortada.
      x_inicial (int): Coordenada x inicial del área de recorte.
      x_final (int): Coordenada x final del área de recorte.
      y_inicial (int): Coordenada y inicial del área de recorte.
      y_final (int): Coordenada y final del área de recorte.

    Return
      None
    """
    try:
        # Abrir la imagen
        image = cv2.imread(ruta_img)
        
        # imprimir el tamaño de la imagen original
        print("El tamaño de la imagen original es" + str(image.shape))

        # Obtener la imagen recortada
        image_crop = image[x_inicial:x_final, y_inicial:y_final]

        # Guardar la imagen recortada en la ruta indicada
        cv2.imwrite(ruta_img_crop, image_crop)

        print("Imagen recortada con éxito. El tamaño de la imagen es de" + str(image_crop.shape))
        
        # Mostrar la imagen recortada
        # Convertir la imagen de BGR a RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.show()
        
    except Exception as e:
        print("Ha ocurrido un error:", str(e))

#Imagenes
recortar_imagen_v2("Imagen1/imagen1.png", "Imagen1/imagen1Cortada.png", 0, 300, 0, 300)
recortar_imagen_v2("Imagen2/imagen2.jpeg", "Imagen2/imagen2Cortada.jpg", 0, 300, 0, 300)


# parte 4 
imagen1Cortada = plt.imread("Imagen1/imagen1Cortada.png")
m_imagen1Cortada = np.array(imagen1Cortada)
print("\nLa matriz de imagen1Cortada es:")
print(m_imagen1Cortada)
print("\nEl tamaño de la imagen1:" + str(imagen1Cortada.shape) + "\n")


# parte 5 
imagen1Cortada = plt.imread("Imagen1/imagen1Cortada.png")
imagen2Cortada = plt.imread("Imagen2/imagen2Cortada.jpg")

m_imagen1Cortada = np.array(imagen1Cortada)
m_imagen2Cortada = np.array(imagen2Cortada)

print("\nLa matriz de imagen1 es:")
print(m_imagen1Cortada)

print("\nLa matriz de imagen2 es:")
print(m_imagen2Cortada)

imagen1Cortada_t = m_imagen1Cortada.transpose(1, 0, 2)
imagen2Cortada_t = m_imagen2Cortada.transpose(1, 0, 2)

plt.imsave("Imagen1/imagen1Cortada_t.png", imagen1Cortada_t)
plt.title("Imagen1 cortada transpuesta")
plt.imshow(imagen1Cortada_t)
plt.show()

print("\nLa matriz transouesta de 'imagen1Cortada.png es:")
print(imagen1Cortada_t)

plt.imsave("Imagen2/imagen2Cortada_t.jpg", imagen2Cortada_t)
plt.imshow(imagen2Cortada_t)
plt.title("Imagen2 cortada transpuesta")
plt.show()

print("\nLa matriz transouesta de 'magen2Cortada.jpg es:")
print(imagen2Cortada_t)


# parte 6 
'''
np.mean calcula el promedio de los valores de los canales en cada pixel de la imagen.
El parámetro axis=2 indica que se realizará el promedio a lo largo del tercer eje,
que corresponde a los canales R, G y B.
'''
# Convertir a escala de grises usando el promedio de los canales RGB
imagen1_Cortada_gris = np.mean(imagen1Cortada, axis=2)
imagen2_Cortada_gris = np.mean(imagen2Cortada, axis=2)

plt.imsave("Imagen1/imagen1Cortada_gris.png", imagen1_Cortada_gris)
plt.imsave("Imagen2/imagen2Cortada_gris.jpg", imagen2_Cortada_gris)

# Mostrar la imagen en escala de grises
plt.imshow(imagen1_Cortada_gris, cmap='gray')
plt.title("Imagen1 Recortada en grises")
plt.show()

plt.imshow(imagen2_Cortada_gris, cmap='gray')
plt.title("Imagen2 Recortada en grises")
plt.show()


# parte 7 
def verificar_inversa(matriz):
    try:
        # Verificamos la forma de la matriz
        print(f"Forma de la matriz: {matriz.shape}")

        # Comprobamos si la matriz es bidimensional
        if matriz.ndim != 2:
            print(f"La matriz debe ser bidimensional para calcular su inversa. Forma actual: {matriz.shape}")
            print("Redimensionando matriz a 2D.")
            # Redimensionar la matriz a 2D (si tiene forma (300, 300, 3), tomamos solo la primera capa)
            matriz_2d = matriz[:, :, 0] 
        else:
            matriz_2d = matriz  # Si ya es 2D, la usamos tal cual

        # Calcular la inversa de la matriz 2D
        inversa = np.linalg.inv(matriz_2d)
        print("La inversa existe y es:")
        print(inversa)
        
    except np.linalg.LinAlgError:
        print("La matriz no tiene inversa.")

imagen1_Cortada_gris = plt.imread("Imagen1/imagen1Cortada_gris.png")
imagen2_Cortada_gris = plt.imread("Imagen2/imagen2Cortada_gris.jpg")

print("\n Matriz inversa de imagen1 cortada gris:")
verificar_inversa(imagen1_Cortada_gris)

print("\nMatriz inversa de imagen2 cortada gris:")
verificar_inversa(imagen2_Cortada_gris)


# parte 8 
def ajustar_contraste(imagen, escalar):
    # se convierte la imagen a tipo float para evitar problemas de overflow
    imagen_float = imagen.astype(np.float32)

    # Ajustar el contraste
    imagen_ajustada = escalar * imagen_float
    
    # Asegurar que los valores estén en el rango [0, 255]
    imagen_ajustada = np.clip(imagen_ajustada, 0, 1)  

    # Escalar al rango [0, 255]
    imagen_ajustada = (imagen_ajustada * 255).astype(np.uint8)
    
    return imagen_ajustada


# Cargar una de las imágenes recortadas en escala de grises
imagen1_cortada_gris = plt.imread("Imagen1/imagen1Cortada_gris.png")

# Caso 1: escalar > 1 (aumentando el contraste)
escalar1 = 1.5  # Ejemplo de escalar mayor que 1
imagen_contraste_aumentado = ajustar_contraste(imagen1_cortada_gris, escalar1)

# Caso 2: 0 < escalar < 1 (disminuyendo el contraste)
escalar2 = 0.5  # Ejemplo de escalar menor que 1
imagen_contraste_disminuido = ajustar_contraste(imagen1_cortada_gris, escalar2)


# Mostrar las imágenes originales y ajustadas
plt.figure(figsize=(15, 10))

# Imagen original en escala de grises
plt.subplot(1, 3, 1)
plt.imshow(imagen1_cortada_gris, cmap='gray')
plt.title("Imagen Original en Escala de Grises")
plt.axis('off')

# Imagen con contraste aumentado
plt.subplot(1, 3, 2)
plt.imshow(imagen_contraste_aumentado, cmap='gray')
plt.title(f"Contraste Aumentado (Escalar={escalar1})")
plt.axis('off')
plt.imsave("Imagen1/imagen1_contraste_aumentado.png", imagen_contraste_aumentado)

# Imagen con contraste disminuido
plt.subplot(1, 3, 3)
plt.imshow(imagen_contraste_disminuido, cmap='gray')
plt.title(f"Contraste Disminuido (Escalar={escalar2})")
plt.axis('off')
plt.imsave("Imagen1/imagen1_contraste_disminuido.png", imagen_contraste_disminuido)
plt.tight_layout()
plt.show()


# parte 9 
# Multiplicación de matrices (prueba para demostrar que la multiplicación de matrices no es conmutativa, 
# aca se multiplica la matriz w con la imagen original en diferentes órdenes).

def multiplicacion_matrices():
# Crear matriz identidad del mismo tamaño que la imagen
  imagen = plt.imread("Imagen1/imagen1.png")
  W = np.eye(imagen.shape[0])  # Esto crea una matriz identidad
  W = np.fliplr(W)  # Voltea la matriz identidad


  resultado1 = np.dot(W, imagen[:, :, 0])  # Primero W luego la imagen (reduzco un canal de la imagen ya que como es rgb tiene 3 canales, a efectos practicos se demuestra correctamente)
  resultado2 = np.dot(imagen[:, :, 0], W)  # Primero la imagen luego W (reduzco un canal de la imagen ya que como es rgb tiene 3 canales, a efectos practicos se demuestra correctamente)

  # Mostrar resultados
  plt.subplot(1, 2, 1)
  plt.imshow(resultado1)
  plt.title("Resultado W * Imagen")
  plt.axis('off')
  plt.imsave("Imagen1/imagen1_matriz_x_imagen.png", resultado1)

  plt.subplot(1, 2, 2)
  plt.imshow(resultado2)
  plt.title("Resultado Imagen * W")
  plt.axis('off')
  plt.show()
  plt.imsave("Imagen1/imagen1_imagen_x_matriz.png", resultado2)

multiplicacion_matrices()


# parte 10 
imagen = plt.imread("Imagen1/imagen1.png")

# Si la imagen tiene 3 canales (RGB), convertir a escala de grises
if imagen.ndim == 3:
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
else:
    imagen_gris = imagen

# Crear la matriz auxiliar
matriz_auxiliar = np.full(imagen_gris.shape, 255, dtype=np.uint8)

# Calcular el negativo
imagen_negativa = matriz_auxiliar - imagen_gris

# Mostrar la imagen negativa
plt.imshow(imagen_negativa, cmap='gray')
plt.title("Imagen Negativa")
plt.axis('off')
plt.show()
plt.imsave("Imagen1/imagen1_negativa.png", imagen_negativa)









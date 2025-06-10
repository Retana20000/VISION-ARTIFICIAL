# Importar librerías necesarias
import cv2  # OpenCV para procesamiento de imágenes
import numpy as np  # Operaciones numéricas
import matplotlib.pyplot as plt  # Visualización de imágenes

# Cargar la imagen en escala de grises
img = cv2.imread('luffy5.png', cv2.IMREAD_GRAYSCALE)
# Se carga la imagen en escala de grises para facilitar la detección de bordes

# --------- Método 1: Laplaciano ---------
laplaciano = cv2.Laplacian(img, cv2.CV_64F)
# Detecta bordes en todas direcciones aplicando la segunda derivada
laplaciano = cv2.convertScaleAbs(laplaciano)
# Convierte a valores absolutos en escala de 8 bits para visualización

# --------- Método 2: Sobel X ---------
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# Detecta bordes verticales (cambios horizontales en la imagen)
sobelx = cv2.convertScaleAbs(sobelx)
# Convierte a escala de 8 bits

# --------- Método 3: Sobel Y ---------
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
# Detecta bordes horizontales (cambios verticales en la imagen)
sobely = cv2.convertScaleAbs(sobely)
# Convierte a escala de 8 bits

# --------- Método 4: Canny ---------
canny = cv2.Canny(img, 100, 200)
# Aplica el algoritmo de Canny con umbrales de histéresis 100 y 200

# --------- Visualización de resultados ---------
plt.figure(figsize=(10, 6))  # Configura el tamaño del gráfico

# Imagen original
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Resultado Laplaciano
plt.subplot(2, 3, 2)
plt.imshow(laplaciano, cmap='gray')
plt.title('Laplaciano')
plt.axis('off')

# Resultado Sobel X
plt.subplot(2, 3, 3)
plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X')
plt.axis('off')

# Resultado Sobel Y
plt.subplot(2, 3, 4)
plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y')
plt.axis('off')

# Filtro Canny
plt.subplot(2, 3, 5)
plt.imshow(canny, cmap='gray')
plt.title('Canny')
plt.axis('off')

# Ajustamos el diseño para que no se encimen las imágenes
plt.tight_layout()
plt.show()

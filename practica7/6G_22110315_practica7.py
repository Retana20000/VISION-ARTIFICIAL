# Importamos las librerías necesarias
import cv2  # Librería de OpenCV para procesamiento de imágenes
import numpy as np  # Librería para operaciones numéricas
import matplotlib.pyplot as plt  # Librería para mostrar imágenes y gráficas

# Cargar la imagen en escala de grises
img = cv2.imread('luffy5.png', cv2.IMREAD_GRAYSCALE)
# Leemos la imagen 'Samus.png' en escala de grises para facilitar el procesamiento

# -------- Filtro Gaussiano para suavizar el ruido --------
img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
# Aplicamos un filtro gaussiano con kernel de 5x5 para reducir el ruido en la imagen

# -------- Crear un kernel para morfología --------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
# Creamos una estructura rectangular de 9x9 píxeles que se usará en las operaciones morfológicas

# -------- Operaciones morfológicas --------
tophat = cv2.morphologyEx(img_suavizada, cv2.MORPH_TOPHAT, kernel)
# TopHat resalta detalles claros sobre un fondo oscuro (imagen original - apertura)

blackhat = cv2.morphologyEx(img_suavizada, cv2.MORPH_BLACKHAT, kernel)
# BlackHat resalta detalles oscuros sobre un fondo claro (cierre - imagen original)

# -------- Mostrar los resultados --------
plt.figure(figsize=(10, 6))  # Preparamos una figura con tamaño personalizado

# Imagen original
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original: luffy5.png')
plt.axis('off')  # Ocultamos los ejes

# Imagen suavizada
plt.subplot(2, 2, 2)
plt.imshow(img_suavizada, cmap='gray')
plt.title('Suavizado (Filtro Gaussiano)')
plt.axis('off')

# Resultado de TopHat
plt.subplot(2, 2, 3)
plt.imshow(tophat, cmap='gray')
plt.title('TopHat')
plt.axis('off')

# Resultado de BlackHat
plt.subplot(2, 2, 4)
plt.imshow(blackhat, cmap='gray')
plt.title('BlackHat')
plt.axis('off')

# Ajustamos el diseño para que no se encimen las imágenes
plt.tight_layout()
plt.show()  # Mostramos la ventana con todas las imágenes
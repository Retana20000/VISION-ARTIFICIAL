import cv2  # Librería OpenCV para procesamiento de imágenes
import numpy as np  # Librería NumPy para operaciones numéricas con matrices

# --------- Cargar imagen y convertir a escala de grises ---------
img_color = cv2.imread('luffy5.png')  # Cargar imagen en color desde archivo
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # Convertir imagen a escala de grises para análisis más sencillo

# --------- Definir y extraer la Región de Interés (ROI) ---------
x, y, w, h = 450, 330, 100, 100  # Coordenadas iniciales (x, y) y dimensiones (w: ancho, h: alto) del recorte
roi_color = img_color[y:y+h, x:x+w]  # Extraer ROI en color
roi_gray = img_gray[y:y+h, x:x+w]    # Extraer ROI en escala de grises

# --------- Detección de esquinas usando el algoritmo de Harris ---------
roi_gray = np.float32(roi_gray)  # Convertir la imagen a tipo float32, requerido por cornerHarris
dst = cv2.cornerHarris(roi_gray, blockSize=2, ksize=3, k=0.04)
# Parámetros:
# - blockSize: Tamaño del vecindario considerado para cada píxel
# - ksize: Tamaño del kernel de Sobel usado para calcular derivadas
# - k: Parámetro de sensibilidad entre 0.04 y 0.06

dst = cv2.dilate(dst, None)  # Dilatar la imagen para resaltar mejor las esquinas detectadas

# --------- Marcar las esquinas detectadas en la imagen ---------
roi_color[dst > 0.01 * dst.max()] = [0, 0, 255]  # Pintar de rojo (BGR) los píxeles que superan el umbral

# --------- Mostrar el resultado en una ventana ---------
cv2.imshow('ROI con esquinas detectadas', roi_color)  # Mostrar solo el ROI con las esquinas marcadas
cv2.waitKey(0)  # Esperar a que el usuario presione una tecla
cv2.destroyAllWindows()  # Cerrar la ventana de la imagen

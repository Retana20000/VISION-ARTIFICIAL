import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen original
img = cv2.imread('luffy5.png', cv2.IMREAD_GRAYSCALE)

# --- Imagen modificada desde práctica 2 (Ampliada) ---
img_modificada = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# --- Histograma original ---
hist_original = cv2.calcHist([img_modificada], [0], None, [256], [0, 256])

# --- Ecualización ---
img_eq = cv2.equalizeHist(img_modificada)

# --- Histograma ecualizado ---
hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

# --- Mostrar resultados con matplotlib ---
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(img_modificada, cmap='gray')
plt.title('Imagen Modificada (Ampliada)')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_original, color='black')
plt.title('Histograma Original')
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Imagen Ecualizada')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Histograma Ecualizado')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

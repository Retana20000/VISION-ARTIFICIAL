import cv2  # Librería OpenCV para procesamiento de imágenes

# --------- Cargar las imágenes a procesar ---------
img1 = cv2.imread('template.png', 0)  # Cargar la imagen plantilla (a buscar), en escala de grises
img2 = cv2.imread('luffy5.png', 0)    # Cargar la imagen completa donde se realizará la búsqueda, también en grises
# Trabajar en escala de grises facilita el análisis y reduce el costo computacional

# --------- Inicializar el detector ORB ---------
orb = cv2.ORB_create()
# ORB: Oriented FAST and Rotated BRIEF
# Es un detector y descriptor eficiente, rápido y robusto frente a rotación y cambios de escala

# --------- Detectar puntos clave y calcular descriptores ---------
kp1, des1 = orb.detectAndCompute(img1, None)  # Puntos clave y descriptores de la plantilla
kp2, des2 = orb.detectAndCompute(img2, None)  # Puntos clave y descriptores de la imagen principal
# Los keypoints representan regiones importantes; los descriptores codifican su apariencia local

# --------- Crear el objeto Brute-Force Matcher ---------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# NORM_HAMMING: métrica usada para comparar descriptores binarios como los de ORB
# crossCheck=True: la coincidencia se valida en ambos sentidos (mayor precisión)

# --------- Realizar el emparejamiento entre descriptores ---------
matches = bf.match(des1, des2)  # Buscar coincidencias entre descriptores de ambas imágenes

# --------- Ordenar coincidencias según la distancia ---------
matches = sorted(matches, key=lambda x: x.distance)
# Se priorizan las coincidencias con menor distancia (más similares)

# --------- Dibujar las 20 mejores coincidencias encontradas ---------
resultado = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
# Visualizar líneas que conectan los puntos coincidentes entre ambas imágenes

# --------- Mostrar el resultado final ---------
cv2.imshow('Similitudes con ORB', resultado)  # Mostrar la ventana con coincidencias visuales
cv2.waitKey(0)  # Esperar a que el usuario presione una tecla para cerrar
cv2.destroyAllWindows()  # Cerrar todas las ventanas de OpenCV

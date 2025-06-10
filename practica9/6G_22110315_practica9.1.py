import cv2

# Cargar la imagen original en escala de grises
img = cv2.imread('luffy5.png', cv2.IMREAD_GRAYSCALE)

# Coordenadas del ROI que quieras extraer (ajústalas tú)
x, y, w, h = 450, 330, 100, 100
template = img[y:y+h, x:x+w]

# Guardar la plantilla
cv2.imwrite('template.png', template)
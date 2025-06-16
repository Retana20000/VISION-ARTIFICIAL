import cv2  # Librería OpenCV para manejo de video e imágenes

# --------- Captura de video ---------
cap = cv2.VideoCapture(0)  # Inicia la cámara web (puedes usar un archivo de video: 'video.mp4')
# El número 0 indica la cámara por defecto; si usas un video, reemplázalo por la ruta del archivo

# --------- Leer el primer frame como fondo de referencia ---------
ret, fondo = cap.read()  # Captura el primer frame
fondo_gray = cv2.cvtColor(fondo, cv2.COLOR_BGR2GRAY)  # Convierte ese frame a escala de grises
# Esta imagen se usará como fondo "estático" para comparar contra los siguientes frames

# --------- Bucle principal ---------
while True:
    ret, frame = cap.read()  # Captura frame por frame en tiempo real
    if not ret:
        break  # Si no se pudo leer el frame, se sale del bucle

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el frame actual a escala de grises

    # --------- Detección de movimiento ---------
    diff = cv2.absdiff(fondo_gray, gray)  # Calcula la diferencia absoluta entre el fondo y el frame actual

    # --------- Aplicar umbral ---------
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    # Los cambios mayores al umbral (30) se convierten en blanco (255), el resto en negro
    # Esto resalta las zonas donde ocurrió movimiento

    # --------- Mostrar los resultados ---------
    cv2.imshow('Video Original', frame)         # Muestra el video original en color
    cv2.imshow('Movimiento detectado', thresh)  # Muestra las zonas con movimiento detectado

    # --------- Salir al presionar ESC ---------
    if cv2.waitKey(1) & 0xFF == 27:  # Espera una tecla cada 1 ms; 27 es el código ASCII de ESC
        break

# --------- Liberar recursos ---------
cap.release()  # Libera la cámara o archivo de video
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV

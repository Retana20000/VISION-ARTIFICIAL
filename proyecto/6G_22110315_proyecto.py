import cv2
import numpy as np
from keras.models import load_model

# Cargar modelo previamente entrenado
modelo = load_model("modelo_emociones_fer2013.h5")

# Lista de emociones del dataset FER2013
emociones = ['Enojo', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Cargar clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iniciar cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y+h, x:x+w]
        rostro_redimensionado = cv2.resize(rostro, (48, 48))
        rostro_normalizado = rostro_redimensionado / 255.0
        entrada = rostro_normalizado.reshape(1, 48, 48, 1)

        prediccion = modelo.predict(entrada)
        emocion = emociones[np.argmax(prediccion)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Reconocimiento de Emociones", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


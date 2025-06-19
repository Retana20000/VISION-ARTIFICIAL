import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from datasets import load_dataset
from PIL import Image
import io

# 🔄 Cargar dataset FER-2013
print("🔄 Cargando dataset FER‑2013...")
dataset = load_dataset("Jeneral/fer-2013", split="train")

# 🧹 Decodificar y procesar imágenes
print("🧹 Decodificando imágenes JPEG...")
X = []
y = []

for sample in dataset:
    try:
        img = Image.open(io.BytesIO(sample["img_bytes"])).convert("L")  # Escala de grises
        img = img.resize((48, 48))
        arr = np.array(img).reshape(48, 48, 1) / 255.0
        X.append(arr)
        y.append(sample["labels"])
    except:
        continue

X = np.stack(X)
y = to_categorical(y, num_classes=7)
print(f"✅ Imágenes cargadas: {X.shape}, Labels: {y.shape}")

# 🧠 Crear modelo
print("🧠 Construyendo modelo...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Entrenar modelo
print("🚀 Entrenando modelo...")
model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)

# 💾 Guardar modelo
print("💾 Guardando modelo...")
model.save("modelo_emociones_fer2013.h5")
print("✅ Modelo guardado como modelo_emociones_fer2013.h5")

import numpy as np
from datasets import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("🔄 Cargando dataset FER‑2013...")
dataset = load_dataset("Jeneral/fer2013", split="train")

print("🧹 Filtrando imágenes inválidas...")
valid_samples = [sample for sample in dataset if len(sample["img_bytes"]) == 2304]

X = np.stack([
    np.frombuffer(sample["img_bytes"], dtype=np.uint8).reshape(48, 48, 1) / 255.0
    for sample in valid_samples
])
y = to_categorical([sample["labels"] for sample in valid_samples], num_classes=7)

# Dividir datos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición del modelo
print("🔧 Creando modelo CNN...")
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(7,activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 Entrenando modelo...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val))

print("💾 Guardando modelo en 'modelo_emociones_fer2013.h5'...")
model.save("modelo_emociones_fer2013.h5")
print("✅ Modelo entrenado y guardado correctamente, mi amor.")

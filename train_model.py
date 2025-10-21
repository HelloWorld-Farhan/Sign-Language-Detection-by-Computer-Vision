import os
import json
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ===== SETTINGS =====
IMG_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 32
RAW_DATA_DIR = 'data'  # folder with subfolders A,B,C,D,E

# ===== INIT MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ===== LOAD AND PROCESS DATA =====
X, y = [], []
label_map = {}
categories = sorted(os.listdir(RAW_DATA_DIR))

for idx, category in enumerate(categories):
    label_map[idx] = category
    category_path = os.path.join(RAW_DATA_DIR, category)
    for file_name in os.listdir(category_path):
        img_path = os.path.join(category_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            h, w, _ = img.shape
            x_coords = [lm.x for lm in result.multi_hand_landmarks[0].landmark]
            y_coords = [lm.y for lm in result.multi_hand_landmarks[0].landmark]
            x_min = max(int(min(x_coords) * w) - 20, 0)
            y_min = max(int(min(y_coords) * h) - 20, 0)
            x_max = min(int(max(x_coords) * w) + 20, w)
            y_max = min(int(max(y_coords) * h) + 20, h)

            roi = img[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            X.append(roi_resized)
            y.append(idx)

X = np.array(X, dtype="float32") / 255.0
y = to_categorical(np.array(y), num_classes=len(categories))

# Save label map
with open("label_map.json", "w") as f:
    json.dump(label_map, f)
print("✅ Label map saved.")

# ===== SPLIT DATA =====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# ===== BUILD MODEL =====
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ===== TRAIN =====
checkpoint = ModelCheckpoint(
    'model.h5',              # ✅ Save as .h5 file
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max'
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

# ===== FINAL SAVE (ensures last model is also stored) =====
model.save("model_final.h5")

print("✅ Training complete. Best model saved to model.h5, final model saved to model_final.h5")

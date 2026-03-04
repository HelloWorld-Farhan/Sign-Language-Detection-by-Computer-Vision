import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import mediapipe as mp

print("=" * 80)
print("🎯 ISL TRAINING - BACKGROUND REMOVAL + HAND FOCUS")
print("=" * 80 + "\n")

# ===================== CONFIG =====================
IMG_SIZE = 224
BATCH_SIZE = 24
EPOCHS = 80
DATASET_PATH = "dataset"
OUTPUT_DATASET = "dataset_no_background"

# ===================== STEP 1: REMOVE BACKGROUNDS =====================
print("🔧 Step 1: Removing backgrounds from all images...\n")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
)


def remove_background(image_path):
    """Remove background and keep only hand"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = img_rgb.shape[:2]

            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            padding = 40
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            mask[y_min:y_max, x_min:x_max] = 255

            result = np.ones_like(img_rgb) * 255
            result[mask == 255] = img_rgb[mask == 255]

            return Image.fromarray(result)

        else:
            return Image.fromarray(img_rgb)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


# Create output directory
if not os.path.exists(OUTPUT_DATASET):
    os.makedirs(OUTPUT_DATASET)

total_processed = 0
total_failed = 0

for letter in sorted(os.listdir(DATASET_PATH)):
    letter_input_path = os.path.join(DATASET_PATH, letter)

    if not os.path.isdir(letter_input_path):
        continue

    letter_output_path = os.path.join(OUTPUT_DATASET, letter)
    if not os.path.exists(letter_output_path):
        os.makedirs(letter_output_path)

    images = [
        f
        for f in os.listdir(letter_input_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {letter}: {len(images)} images...")

    for img_name in images:
        img_path = os.path.join(letter_input_path, img_name)
        output_path = os.path.join(letter_output_path, img_name)

        if os.path.exists(output_path):
            total_processed += 1
            continue

        result_img = remove_background(img_path)

        if result_img:
            result_img.save(output_path)
            total_processed += 1
        else:
            total_failed += 1

hands.close()

print("\n✅ Background removal complete!")
print(f"   Processed: {total_processed}")
print(f"   Failed: {total_failed}\n")

# ===================== STEP 2: TRAIN ON CLEAN DATA =====================
print("=" * 80)
print("🎯 Step 2: Training on background-removed images")
print("=" * 80 + "\n")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.15,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.25,
    zoom_range=[0.7, 1.3],
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    fill_mode="constant",
    cval=255,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.15)

train_data = train_datagen.flow_from_directory(
    OUTPUT_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_data = val_datagen.flow_from_directory(
    OUTPUT_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

NUM_CLASSES = train_data.num_classes
print(f"✅ Training: {train_data.samples} | Validation: {val_data.samples}\n")

# ===================== BUILD MODEL =====================
print("🏗️  Building model...\n")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=1.0,
)

for layer in base_model.layers[: int(len(base_model.layers) * 0.7)]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.005))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"Parameters: {model.count_params():,}\n")

# ===================== CALLBACKS =====================
callbacks = [
    ModelCheckpoint(
        "isl_no_background_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    ),
]

# ===================== TRAINING =====================
print("🚀 Starting training...\n")

history = model.fit(
    train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# ===================== SAVE =====================
model.save("isl_no_background_final.h5")
print("\n✅ Model saved\n")

# ===================== CONVERT TO TFLITE =====================
print("Converting to TFLite...\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("isl_no_background.tflite", "wb") as f:
    f.write(tflite_model)

print(
    f"✅ TFLite saved: isl_no_background.tflite ({len(tflite_model)/1024/1024:.2f} MB)\n"
)

# ===================== EVALUATE =====================
val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f"Final Validation Accuracy: {val_acc*100:.2f}%\n")

# ===================== PLOT =====================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy (No Background)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss (No Background)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_no_background.png", dpi=150)

print("✅ Plot saved\n")

print("=" * 80)
print("🎉 COMPLETE!")
print("=" * 80)

print("\nFiles created:")
print("  1. dataset_no_background/ - Cleaned dataset")
print("  2. isl_no_background.tflite - Model for Android")
print("  3. training_no_background.png - Training plot")
print("\nNext step: Copy isl_no_background.tflite to your app!\n")

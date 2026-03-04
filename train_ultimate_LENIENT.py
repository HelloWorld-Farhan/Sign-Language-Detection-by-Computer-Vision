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
print("🎯 ISL ULTIMATE TRAINING - LENIENT MODE")
print("=" * 80 + "\n")

# ===================== CONFIG =====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 100
DATASET_PATH = "dataset"
OUTPUT_DATASET = "dataset_clean"

# ===================== STEP 1: LENIENT BACKGROUND REMOVAL =====================
print("🔧 STEP 1: Processing images with LENIENT hand detection...\n")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.1,  # VERY LENIENT!
    min_tracking_confidence=0.1,
)


def remove_background_lenient(image_path):
    """LENIENT: Try multiple approaches to detect and isolate hand"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Try hand detection
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Hand detected! Remove background
            hand_landmarks = results.multi_hand_landmarks[0]

            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            # Add generous padding
            padding = 80
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_max = min(h, max(y_coords) + padding)

            # Create white background
            white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

            # Copy hand region
            white_bg[y_min:y_max, x_min:x_max] = img_rgb[y_min:y_max, x_min:x_max]

            return Image.fromarray(white_bg)

        else:
            # No hand detected - Use fallback: Simple color-based segmentation
            # This works for most hand images with different backgrounds

            # Convert to HSV
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

            # Skin color range (works for most skin tones)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

            lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
            upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

            # Combine masks
            mask = cv2.bitwise_or(mask1, mask2)

            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find largest contour (assume it's the hand)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Get bounding box
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)

                # Add padding
                padding = 30
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_box = min(w - x, w_box + 2 * padding)
                h_box = min(h - y, h_box + 2 * padding)

                # Create white background
                white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

                # Copy hand region
                white_bg[y : y + h_box, x : x + w_box] = img_rgb[
                    y : y + h_box, x : x + w_box
                ]

                return Image.fromarray(white_bg)

            else:
                # Fallback: Just use center crop on white background
                # This is better than rejecting the image
                white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255

                # Center crop (assume hand is in center)
                crop_size = min(h, w)
                start_x = (w - crop_size) // 2
                start_y = (h - crop_size) // 2

                white_bg[
                    start_y : start_y + crop_size, start_x : start_x + crop_size
                ] = img_rgb[
                    start_y : start_y + crop_size, start_x : start_x + crop_size
                ]

                return Image.fromarray(white_bg)

    except Exception as e:
        print(f"   ❌ Error: {os.path.basename(image_path)} - {e}")
        return None


# Create output directory
if not os.path.exists(OUTPUT_DATASET):
    os.makedirs(OUTPUT_DATASET)

total_processed = 0
total_failed = 0
hand_detected = 0
fallback_used = 0

print("Processing all images...\n")

for letter in sorted(os.listdir(DATASET_PATH)):
    letter_input = os.path.join(DATASET_PATH, letter)
    if not os.path.isdir(letter_input):
        continue

    letter_output = os.path.join(OUTPUT_DATASET, letter)
    if not os.path.exists(letter_output):
        os.makedirs(letter_output)

    images = [
        f
        for f in os.listdir(letter_input)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"📁 {letter}: Processing {len(images)} images...", end=" ")

    letter_processed = 0
    letter_failed = 0

    for img_name in images:
        input_path = os.path.join(letter_input, img_name)
        output_path = os.path.join(letter_output, img_name)

        # Skip if already exists
        if os.path.exists(output_path):
            letter_processed += 1
            total_processed += 1
            continue

        # Process
        result_img = remove_background_lenient(input_path)

        if result_img:
            result_img.save(output_path, "JPEG", quality=95)
            letter_processed += 1
            total_processed += 1
        else:
            letter_failed += 1
            total_failed += 1

    print(f"✅ {letter_processed} done, {letter_failed} failed")

hands.close()

print(f"\n{'=' * 80}")
print(f"✅ Processing complete!")
print(f"   Total processed: {total_processed}")
print(f"   Total failed: {total_failed}")
print(f"{'=' * 80}\n")

# ===================== VERIFY DATASET =====================
print("🔍 Verifying dataset...\n")

class_counts = {}
for letter in sorted(os.listdir(OUTPUT_DATASET)):
    letter_path = os.path.join(OUTPUT_DATASET, letter)
    if os.path.isdir(letter_path):
        count = len(
            [
                f
                for f in os.listdir(letter_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        class_counts[letter] = count
        print(f"   {letter}: {count} images")

total_images = sum(class_counts.values())
print(f"\n✅ Total images: {total_images}\n")

if total_images < 500:
    print("⚠️ WARNING: Low image count. Training may not be optimal.")
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != "yes":
        exit(0)

# ===================== AUGMENTATION =====================
print("=" * 80)
print("🔧 Setting up augmentation")
print("=" * 80 + "\n")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=[0.8, 1.2],
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="constant",
    cval=1.0,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# ===================== LOAD DATA =====================
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
print("=" * 80)
print("🏗️  Building model")
print("=" * 80 + "\n")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=1.0,
)

for layer in base_model.layers[: int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(
    NUM_CLASSES, activation="softmax", kernel_regularizer=regularizers.l2(0.005)
)(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"✅ Model: {model.count_params():,} parameters\n")

# ===================== CALLBACKS =====================
callbacks = [
    ModelCheckpoint(
        "isl_clean_best.h5",
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

# ===================== TRAIN =====================
print("=" * 80)
print("🚀 TRAINING Phase 1")
print("=" * 80 + "\n")

history = model.fit(
    train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# ===================== FINE-TUNE =====================
print("\n" + "=" * 80)
print("🎯 TRAINING Phase 2 (Fine-tuning)")
print("=" * 80 + "\n")

for layer in base_model.layers[int(len(base_model.layers) * 0.8) :]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00003),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    initial_epoch=len(history.history["loss"]),
    callbacks=callbacks,
    verbose=1,
)

# ===================== SAVE =====================
model.save("isl_clean_final.h5")

# ===================== EVALUATE =====================
val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f"\n✅ Final Validation Accuracy: {val_acc * 100:.2f}%\n")

# ===================== CONVERT TFLITE =====================
print("📱 Converting to TFLite...\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def rep_data():
    for _ in range(100):
        yield [next(train_data)[0].astype(np.float32)]


converter.representative_dataset = rep_data
tflite_model = converter.convert()

with open("isl_clean.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite: isl_clean.tflite ({len(tflite_model) / 1024 / 1024:.2f} MB)\n")

# ===================== PLOT =====================
hist_all = {
    "accuracy": history.history["accuracy"] + history_ft.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
    + history_ft.history["val_accuracy"],
    "loss": history.history["loss"] + history_ft.history["loss"],
    "val_loss": history.history["val_loss"] + history_ft.history["val_loss"],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(hist_all["accuracy"], label="Train")
ax1.plot(hist_all["val_accuracy"], label="Val")
ax1.axhline(0.9, color="g", linestyle="--", alpha=0.5)
ax1.set_title("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(hist_all["loss"], label="Train")
ax2.plot(hist_all["val_loss"], label="Val")
ax2.set_title("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_clean.png", dpi=150)

print("=" * 80)
print("🎉 COMPLETE!")
print("=" * 80)
print("\n✅ Files:")
print("   • isl_clean.tflite ← USE THIS IN APP")
print("   • isl_clean_final.h5")
print("   • training_clean.png\n")

if val_acc >= 0.90:
    print("🏆 EXCELLENT! 90%+ accuracy!")
elif val_acc >= 0.85:
    print("✅ VERY GOOD! 85%+ accuracy!")
else:
    print("⚠️ Needs improvement")

print("\n" + "=" * 80 + "\n")

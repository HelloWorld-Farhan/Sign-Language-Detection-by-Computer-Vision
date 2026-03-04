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
import shutil
import cv2
from PIL import Image
import mediapipe as mp

print("=" * 80)
print("🎯 ISL ULTIMATE TRAINING - MERGED + BACKGROUND REMOVAL")
print("=" * 80 + "\n")

# ===================== CONFIG =====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 80

# Input folders
DATASET_COPY = "dataset"  # Your ISL dataset (500-650 per letter)
ASL_ALPHABET = "asl_alphabet_train"  # ASL dataset (3000 per letter)

# Output folders
MERGE_DATASET = "TheMergeDataSet"  # Step 1: Merge both datasets
NEW_DATASET = "TheNewDataSet"  # Step 2: Add background-removed versions

# ===================== STEP 1: MERGE DATASETS =====================
print("=" * 80)
print("📁 STEP 1: Merging dataset-copy + asl_alphabet_train")
print("=" * 80 + "\n")

if os.path.exists(MERGE_DATASET):
    print(f"   Removing existing {MERGE_DATASET}/...")
    shutil.rmtree(MERGE_DATASET)

os.makedirs(MERGE_DATASET)

# Create A-Z folders
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(os.path.join(MERGE_DATASET, letter), exist_ok=True)

# Merge dataset-copy
print("Merging dataset-copy...")
total_isl = 0

if os.path.exists(DATASET_COPY):
    for letter in sorted(os.listdir(DATASET_COPY)):
        src_path = os.path.join(DATASET_COPY, letter)
        if not os.path.isdir(src_path):
            continue

        # Only A-Z
        if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue

        dst_path = os.path.join(MERGE_DATASET, letter)

        images = [
            f
            for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        count = 0
        for idx, img_name in enumerate(images):
            src_file = os.path.join(src_path, img_name)
            dst_file = os.path.join(dst_path, f"isl_{idx:04d}.jpg")

            try:
                img = Image.open(src_file).convert("RGB")
                img.save(dst_file, "JPEG", quality=95)
                count += 1
                total_isl += 1
            except:
                pass

        print(f"   {letter}: {count} ISL images")
else:
    print("   ⚠️  dataset-copy not found, skipping...")

print(f"\n✅ ISL images merged: {total_isl}\n")

# Merge asl_alphabet_train (limit to 800 per letter to balance)
print("Merging asl_alphabet_train...")
total_asl = 0
ASL_LIMIT = 800  # Balance with ISL data

if os.path.exists(ASL_ALPHABET):
    for letter in sorted(os.listdir(ASL_ALPHABET)):
        src_path = os.path.join(ASL_ALPHABET, letter)
        if not os.path.isdir(src_path):
            continue

        # Only A-Z
        if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue

        dst_path = os.path.join(MERGE_DATASET, letter)

        images = [
            f
            for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Limit ASL images
        images = images[:ASL_LIMIT]

        count = 0
        for idx, img_name in enumerate(images):
            src_file = os.path.join(src_path, img_name)
            dst_file = os.path.join(dst_path, f"asl_{idx:04d}.jpg")

            try:
                img = Image.open(src_file).convert("RGB")
                img.save(dst_file, "JPEG", quality=95)
                count += 1
                total_asl += 1
            except:
                pass

        print(f"   {letter}: {count} ASL images (limited to {ASL_LIMIT})")
else:
    print("   ⚠️  asl_alphabet_train not found, skipping...")

print(f"\n✅ ASL images merged: {total_asl}")
print(f"✅ Total in TheMergeDataSet: {total_isl + total_asl}\n")

# ===================== STEP 2: INITIALIZE BACKGROUND REMOVAL =====================
print("=" * 80)
print("🔧 STEP 2: Initializing background removal")
print("=" * 80 + "\n")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.2
)


def remove_background(image_path):
    """Remove background and return hand on white"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]

            padding = 60
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(w, max(x_coords) + padding)
            y_max = min(h, max(y_coords) + padding)

            white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
            white_bg[y_min:y_max, x_min:x_max] = img_rgb[y_min:y_max, x_min:x_max]
            return Image.fromarray(white_bg)

        else:
            # Fallback: skin detection
            hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest)

                padding = 30
                x = max(0, x - padding)
                y = max(0, y - padding)
                w_box = min(w - x, w_box + 2 * padding)
                h_box = min(h - y, h_box + 2 * padding)

                white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
                white_bg[y : y + h_box, x : x + w_box] = img_rgb[
                    y : y + h_box, x : x + w_box
                ]
                return Image.fromarray(white_bg)
            else:
                return Image.fromarray(img_rgb)

    except:
        return None


# ===================== STEP 3: CREATE THENEWDATASET =====================
print("=" * 80)
print("📁 STEP 3: Creating TheNewDataSet (with background removal)")
print("=" * 80 + "\n")

if os.path.exists(NEW_DATASET):
    print(f"   Removing existing {NEW_DATASET}/...")
    shutil.rmtree(NEW_DATASET)

os.makedirs(NEW_DATASET)

total_original = 0
total_clean = 0

for letter in sorted(os.listdir(MERGE_DATASET)):
    src_path = os.path.join(MERGE_DATASET, letter)
    if not os.path.isdir(src_path):
        continue

    dst_path = os.path.join(NEW_DATASET, letter)
    os.makedirs(dst_path, exist_ok=True)

    images = [
        f for f in os.listdir(src_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Processing {letter}: {len(images)} images...")

    letter_orig = 0
    letter_clean = 0

    for idx, img_name in enumerate(images):
        src_file = os.path.join(src_path, img_name)

        # 1. Copy original (with background)
        dst_original = os.path.join(dst_path, f"orig_{idx:04d}.jpg")
        try:
            img = Image.open(src_file).convert("RGB")
            img.save(dst_original, "JPEG", quality=95)
            letter_orig += 1
            total_original += 1
        except:
            pass

        # 2. Create background-removed version
        clean_img = remove_background(src_file)
        if clean_img:
            dst_clean = os.path.join(dst_path, f"clean_{idx:04d}.jpg")
            clean_img.save(dst_clean, "JPEG", quality=95)
            letter_clean += 1
            total_clean += 1

    print(
        f"   ✅ {letter}: {letter_orig} original + {letter_clean} clean = {letter_orig + letter_clean} total"
    )

hands.close()

print(f"\n✅ TheNewDataSet created!")
print(f"   Original: {total_original}")
print(f"   Clean:    {total_clean}")
print(f"   Total:    {total_original + total_clean}\n")

# ===================== STEP 4: ANALYZE FINAL DATASET =====================
print("=" * 80)
print("📊 STEP 4: Analyzing TheNewDataSet")
print("=" * 80 + "\n")

class_counts = {}
for letter in sorted(os.listdir(NEW_DATASET)):
    letter_path = os.path.join(NEW_DATASET, letter)
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
print(f"\n📊 Statistics:")
print(f"   Total:   {total_images}")
print(f"   Average: {total_images // len(class_counts)}\n")

# ===================== STEP 5: DATA GENERATORS =====================
print("=" * 80)
print("🔧 STEP 5: Setting up augmentation")
print("=" * 80 + "\n")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=[0.85, 1.15],
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    NEW_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_data = val_datagen.flow_from_directory(
    NEW_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

NUM_CLASSES = train_data.num_classes
print(f"✅ Training:   {train_data.samples}")
print(f"✅ Validation: {val_data.samples}")
print(f"✅ Classes:    {NUM_CLASSES}\n")

# ===================== STEP 6: BUILD MODEL =====================
print("=" * 80)
print("🏗️  STEP 6: Building model")
print("=" * 80 + "\n")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=1.0,
)

for layer in base_model.layers[: int(len(base_model.layers) * 0.75)]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(384, activation="relu", kernel_regularizer=regularizers.l2(0.008))(x)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
x = Dense(192, activation="relu", kernel_regularizer=regularizers.l2(0.008))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(
    NUM_CLASSES, activation="softmax", kernel_regularizer=regularizers.l2(0.004)
)(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=0.0004),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"✅ Model: {model.count_params():,} parameters\n")

# ===================== STEP 7: CALLBACKS =====================
callbacks = [
    ModelCheckpoint(
        "isl_ultimate_best.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.001,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
    ),
]

# ===================== STEP 8: TRAIN PHASE 1 =====================
print("=" * 80)
print("🚀 STEP 8: Training Phase 1")
print("=" * 80 + "\n")

history_p1 = model.fit(
    train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks, verbose=1
)

# ===================== STEP 9: FINE-TUNE PHASE 2 =====================
print("\n" + "=" * 80)
print("🎯 STEP 9: Fine-tuning Phase 2")
print("=" * 80 + "\n")

for layer in base_model.layers[int(len(base_model.layers) * 0.75) :]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=0.00004),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_p2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    initial_epoch=len(history_p1.history["loss"]),
    callbacks=callbacks,
    verbose=1,
)

# ===================== STEP 10: SAVE =====================
model.save("isl_ultimate_final.h5")

# ===================== STEP 11: EVALUATE =====================
print("\n" + "=" * 80)
print("📊 EVALUATION")
print("=" * 80 + "\n")

val_loss, val_acc = model.evaluate(val_data, verbose=0)
print(f"🎯 Validation Accuracy: {val_acc * 100:.2f}%")
print(f"📉 Validation Loss:     {val_loss:.4f}\n")

val_data.reset()
predictions = model.predict(val_data, verbose=0)
y_pred = np.argmax(predictions, axis=1)
y_true = val_data.classes

class_labels = list(train_data.class_indices.keys())
class_correct = {l: 0 for l in class_labels}
class_total = {l: 0 for l in class_labels}

for true_idx, pred_idx in zip(y_true, y_pred):
    label = class_labels[true_idx]
    class_total[label] += 1
    if true_idx == pred_idx:
        class_correct[label] += 1

excellent = []
good = []
needs_work = []

print("PER-LETTER ACCURACY:")
print("-" * 60)

for letter in sorted(class_labels):
    if class_total[letter] > 0:
        acc = (class_correct[letter] / class_total[letter]) * 100
        if acc >= 90:
            status = "✅"
            excellent.append(letter)
        elif acc >= 80:
            status = "⚠️"
            good.append(letter)
        else:
            status = "❌"
            needs_work.append(letter)
        print(
            f"{status} {letter}: {acc:.1f}% ({class_correct[letter]}/{class_total[letter]})"
        )

print(f"\n{'=' * 60}")
print(f"✅ Excellent (≥90%):  {len(excellent)}/{NUM_CLASSES}")
print(f"⚠️  Good (80-89%):    {len(good)}/{NUM_CLASSES}")
print(f"❌ Needs work (<80%): {len(needs_work)}/{NUM_CLASSES}\n")

# ===================== STEP 12: CONVERT TFLITE =====================
print("📱 Converting to TFLite...\n")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def rep_data():
    for _ in range(100):
        yield [next(train_data)[0].astype(np.float32)]


converter.representative_dataset = rep_data
tflite_model = converter.convert()

with open("isl_ultimate.tflite", "wb") as f:
    f.write(tflite_model)

file_size = len(tflite_model) / 1024 / 1024
print(f"✅ TFLite: isl_ultimate.tflite ({file_size:.2f} MB)\n")

# ===================== STEP 13: PLOT =====================
hist = {
    "accuracy": history_p1.history["accuracy"] + history_p2.history["accuracy"],
    "val_accuracy": history_p1.history["val_accuracy"]
    + history_p2.history["val_accuracy"],
    "loss": history_p1.history["loss"] + history_p2.history["loss"],
    "val_loss": history_p1.history["val_loss"] + history_p2.history["val_loss"],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(hist["accuracy"], label="Train", linewidth=2, color="#2196F3")
ax1.plot(hist["val_accuracy"], label="Val", linewidth=2, color="#FF9800")
ax1.axhline(0.9, color="g", linestyle="--", alpha=0.6, label="90%")
ax1.set_title("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(hist["loss"], label="Train", linewidth=2, color="#2196F3")
ax2.plot(hist["val_loss"], label="Val", linewidth=2, color="#FF9800")
ax2.set_title("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_ultimate.png", dpi=150)

# ===================== FINAL REPORT =====================
print("=" * 80)
print("🎉 COMPLETE!")
print("=" * 80 + "\n")

train_acc = hist["accuracy"][-1]
val_acc = hist["val_accuracy"][-1]
gap = (train_acc - val_acc) * 100

print("📁 Files:")
print("   ✅ isl_ultimate.tflite ← USE IN APP")
print("   ✅ isl_ultimate_final.h5")
print("   ✅ training_ultimate.png\n")

print("📊 Performance:")
print(f"   Training:   {train_acc * 100:.2f}%")
print(f"   Validation: {val_acc * 100:.2f}%")
print(f"   Gap:        {gap:.1f}%")
print(f"   Size:       {file_size:.2f} MB\n")

print("📈 Results:")
print(f"   ✅ Excellent:   {len(excellent)}/{NUM_CLASSES}")
print(f"   ⚠️  Good:       {len(good)}/{NUM_CLASSES}")
print(f"   ❌ Needs work: {len(needs_work)}/{NUM_CLASSES}\n")

if val_acc >= 0.90 and gap < 10:
    print("🏆 OUTSTANDING! Production ready!")
elif val_acc >= 0.90:
    print("🏆 EXCELLENT! 90%+ achieved!")
elif val_acc >= 0.85:
    print("✅ VERY GOOD!")
else:
    print("⚠️  Good progress")

print("\n" + "=" * 80 + "\n")

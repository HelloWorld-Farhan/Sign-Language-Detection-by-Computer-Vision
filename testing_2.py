import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
import time
import random

print("=" * 80)
print("🔍 ISL MODEL DIAGNOSTIC TOOL - FIND ACCURACY PROBLEMS")
print("=" * 80 + "\n")

MODEL_PATH = "isl_ultimate_90.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

print("✅ Model loaded\n")

# =============================================================================
# TEST 1: BASELINE - Perfect Conditions
# =============================================================================

print("=" * 80)
print("TEST 1: BASELINE - Perfect Conditions")
print("=" * 80 + "\n")


def test_baseline():
    correct = 0
    total = 0

    for letter in labels:
        test_dir = f"dataset/{letter}"

        if not os.path.exists(test_dir):
            continue

        images = (
            glob.glob(f"{test_dir}/*.jpg")
            + glob.glob(f"{test_dir}/*.jpeg")
            + glob.glob(f"{test_dir}/*.png")
        )

        for img_path in images[:10]:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]["index"], img_batch)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]["index"])[0]

            predicted = labels[np.argmax(predictions)]

            if predicted == letter:
                correct += 1

            total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✅ Baseline Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


baseline_acc = test_baseline()

# =============================================================================
# TEST 2: REAL-WORLD SIMULATION
# =============================================================================

print("\n" + "=" * 80)
print("TEST 2: REAL-WORLD SIMULATION")
print("=" * 80 + "\n")


def apply_real_world_effects(img, effect_type):

    if effect_type == "motion_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))

    elif effect_type == "dark":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(0.5)

    elif effect_type == "bright":
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(1.5)

    elif effect_type == "low_contrast":
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.5)

    elif effect_type == "rotation":
        return img.rotate(random.randint(-15, 15))

    elif effect_type == "noise":
        img_array = np.array(img)
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        noisy = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)

    return img


effects = ["motion_blur", "dark", "bright", "low_contrast", "rotation", "noise"]
effect_results = {}

for effect in effects:

    print(f"Testing: {effect}...")
    correct = 0
    total = 0

    for letter in labels:
        test_dir = f"dataset/{letter}"

        if not os.path.exists(test_dir):
            continue

        images = (
            glob.glob(f"{test_dir}/*.jpg")
            + glob.glob(f"{test_dir}/*.jpeg")
            + glob.glob(f"{test_dir}/*.png")
        )

        for img_path in images[:5]:
            img = Image.open(img_path).convert("RGB")
            img = apply_real_world_effects(img, effect)
            img = img.resize((224, 224))

            img_array = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]["index"], img_batch)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]["index"])[0]

            predicted = labels[np.argmax(predictions)]

            if predicted == letter:
                correct += 1

            total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    effect_results[effect] = accuracy

    drop = baseline_acc - accuracy
    print(f"   {effect}: {accuracy:.1f}% (drop: {drop:.1f}%)\n")

# =============================================================================
# TEST 3: CONFUSION ANALYSIS
# =============================================================================

print("=" * 80)
print("TEST 3: CONFUSION ANALYSIS")
print("=" * 80 + "\n")

confusion_pairs = {}

for letter in labels:
    test_dir = f"dataset/{letter}"

    if not os.path.exists(test_dir):
        continue

    images = (
        glob.glob(f"{test_dir}/*.jpg")
        + glob.glob(f"{test_dir}/*.jpeg")
        + glob.glob(f"{test_dir}/*.png")
    )

    for img_path in images[:10]:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]["index"], img_batch)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        predicted = labels[np.argmax(predictions)]

        if predicted != letter:
            pair = f"{letter}→{predicted}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

print("Top 10 Confusions:")

sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[
    :10
]

for pair, count in sorted_confusions:
    print(f"   {pair}: {count} times")

# =============================================================================
# FINAL DIAGNOSIS
# =============================================================================

print("\n" + "=" * 80)
print("🎯 FINAL DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80 + "\n")

print("📊 SUMMARY:")
print(f"   Baseline (perfect): {baseline_acc:.1f}%")
print(f"   Worst real-world: {min(effect_results.values()):.1f}%")
print(
    f"   Average drop: {baseline_acc - np.mean(list(effect_results.values())):.1f}%\n"
)

print("🔴 MAIN PROBLEMS:")

worst_effect = min(effect_results.items(), key=lambda x: x[1])

print(
    f"   1. Biggest accuracy drop: "
    f"{worst_effect[0]} "
    f"({baseline_acc - worst_effect[1]:.1f}% drop)"
)

if sorted_confusions:
    print(
        f"   2. Most confused: "
        f"{sorted_confusions[0][0]} "
        f"({sorted_confusions[0][1]} errors)"
    )

print("\n" + "=" * 80 + "\n")

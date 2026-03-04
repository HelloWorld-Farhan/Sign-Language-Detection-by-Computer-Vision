import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

print("=" * 80)
print("🔬 ULTIMATE MODEL DIAGNOSTIC TEST")
print("=" * 80 + "\n")

# ===================== CONFIG =====================
MODEL_PATH = "isl_clean.tflite"
DATASET_PATH = "dataset_clean"
IMG_SIZE = 224
TEST_SAMPLES_PER_CLASS = 50


# ===================== LOAD MODEL =====================
print("📥 Loading TFLite model...\n")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("✅ Model loaded successfully!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Model size: {os.path.getsize(MODEL_PATH)/(1024*1024):.2f} MB\n")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)


# ===================== LOAD LABELS =====================
class_names = sorted(
    [
        d
        for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ]
)

NUM_CLASSES = len(class_names)

print(f"📋 Found {NUM_CLASSES} classes:")
print(", ".join(class_names), "\n")


# ===================== PREPARE TEST DATA =====================
print("📊 Preparing test data...\n")

test_images = []
test_labels = []

for class_idx, class_name in enumerate(class_names):
    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        f
        for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    test_imgs = (
        images[-TEST_SAMPLES_PER_CLASS:]
        if len(images) >= TEST_SAMPLES_PER_CLASS
        else images
    )

    print(f"{class_name}: Testing {len(test_imgs)} images")

    for img_name in test_imgs:
        img_path = os.path.join(class_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img, dtype=np.float32) / 255.0

            test_images.append(img_array)
            test_labels.append(class_idx)

        except Exception as e:
            print(f"⚠️ Failed to load {img_path}: {e}")

test_images = np.array(test_images)
test_labels = np.array(test_labels)

print("\n✅ Test dataset prepared:")
print(f"Total images: {len(test_images)}")
print(f"Shape: {test_images.shape}\n")


# ===================== RUN PREDICTIONS =====================
print("=" * 80)
print("🚀 RUNNING PREDICTIONS")
print("=" * 80 + "\n")

predictions = []
confidences = []

for i, img in enumerate(test_images):

    if i % 100 == 0:
        print(f"Progress: {i}/{len(test_images)}")

    interpreter.set_tensor(input_details[0]["index"], [img])
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_class = np.argmax(output)
    pred_conf = output[pred_class]

    predictions.append(pred_class)
    confidences.append(pred_conf)

predictions = np.array(predictions)
confidences = np.array(confidences)

print("✅ Predictions complete!\n")


# ===================== OVERALL ACCURACY =====================
correct = np.sum(predictions == test_labels)
accuracy = (correct / len(test_labels)) * 100

print("=" * 80)
print("📊 OVERALL RESULTS")
print("=" * 80 + "\n")

print(f"🎯 Overall Accuracy: {accuracy:.2f}%")
print(f"Correct: {correct}/{len(test_labels)}")
print(f"Incorrect: {len(test_labels)-correct}/{len(test_labels)}\n")

avg_confidence = np.mean(confidences) * 100
print(f"Average Confidence: {avg_confidence:.2f}%\n")


# ===================== PER-CLASS ANALYSIS =====================
per_class_correct = {}
per_class_total = {}
per_class_avg_conf = {}

print("=" * 80)
print("📋 PER-LETTER ACCURACY")
print("=" * 80 + "\n")

for i, class_name in enumerate(class_names):

    mask = test_labels == i

    class_predictions = predictions[mask]
    class_true = test_labels[mask]
    class_confidences = confidences[mask]

    correct_count = np.sum(class_predictions == class_true)
    total_count = len(class_true)

    accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0

    avg_conf = np.mean(class_confidences) * 100 if total_count > 0 else 0

    per_class_correct[class_name] = correct_count
    per_class_total[class_name] = total_count
    per_class_avg_conf[class_name] = avg_conf

    status = "✅" if accuracy_pct >= 90 else "⚠️" if accuracy_pct >= 80 else "❌"

    print(
        f"{status} {class_name}: "
        f"{accuracy_pct:5.1f}% | "
        f"Conf: {avg_conf:5.1f}% | "
        f"{correct_count}/{total_count}"
    )


# ===================== CONFUSION MATRIX =====================
cm = confusion_matrix(test_labels, predictions)

confusion_pairs = []

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i][j] > 0:

            percent = (cm[i][j] / per_class_total[class_names[i]]) * 100

            confusion_pairs.append((class_names[i], class_names[j], cm[i][j], percent))

confusion_pairs.sort(key=lambda x: x[3], reverse=True)

print("\nTop Confusion Pairs:")
for true_label, pred_label, count, percent in confusion_pairs[:10]:
    print(f"{true_label} → {pred_label}: {count} ({percent:.1f}%)")


# ===================== CONFIDENCE ANALYSIS =====================
correct_confidences = confidences[predictions == test_labels]
incorrect_confidences = confidences[predictions != test_labels]

print("\nConfidence Analysis:")
print(f"Correct Avg: {np.mean(correct_confidences)*100:.2f}%")

if len(incorrect_confidences) > 0:
    print(f"Incorrect Avg: {np.mean(incorrect_confidences)*100:.2f}%")
else:
    print("No incorrect predictions 🎉")


# ===================== SAVE REPORT =====================
report = {
    "overall_accuracy": float(accuracy),
    "average_confidence": float(avg_confidence),
    "total_tested": int(len(test_labels)),
}

with open("diagnostic_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n✅ Diagnostic report saved: diagnostic_report.json")


# ===================== FINAL VERDICT =====================
print("\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80 + "\n")

if accuracy >= 90:
    print("🏆 OUTSTANDING! Production ready.")
elif accuracy >= 85:
    print("✅ Very Good! Minor improvements possible.")
elif accuracy >= 80:
    print("⚠️ Good but needs refinement.")
else:
    print("❌ Needs retraining with better data.")

print("\n✅ DIAGNOSTIC COMPLETE!")

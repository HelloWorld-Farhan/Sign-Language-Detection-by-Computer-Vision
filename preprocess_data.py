import os
import cv2
import numpy as np
import random
import json

data_dir = 'data'
categories = ['A', 'B', 'C','D', 'E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

data = []
IMG_SIZE = 64

label_map = {}

for idx, category in enumerate(categories):
    path = os.path.join(data_dir, category)
    class_num = idx
    label_map[str(idx)] = category  # Save mapping for future

    print(f"[{idx+1}/{len(categories)}] Processing {category}...")

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue  # skip corrupted images

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Optional: convert to grayscale for faster training
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = np.expand_dims(img, axis=-1)  # keep shape (64, 64, 1)

            data.append([img, class_num])
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")

# Shuffle and split
random.shuffle(data)
X, y = [], []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X, dtype='float32') / 255.0
y = np.array(y)

# Save
np.save('X.npy', X)
np.save('y.npy', y)

# Save label mapping
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("✅ Preprocessing completed.")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("✅ Label map saved to label_map.json")

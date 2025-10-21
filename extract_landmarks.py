import cv2
import mediapipe as mp
import os
import csv

# Define dataset path
DATASET_DIR = "data"   # <-- Change this if needed
OUTPUT_FILE = "landmarks.csv"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Create CSV header (21 landmarks * 3 coordinates + label)
header = []
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]
header.append("label")

# Create CSV file
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    # Loop through each folder (A–Z)
    for label_folder in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, label_folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing label: {label_folder}")

        # Loop through each image in that folder
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Convert to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # Extract landmarks
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(label_folder)
                writer.writerow(row)

print(f"\n✅ Landmarks saved successfully to {OUTPUT_FILE}")

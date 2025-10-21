import cv2
import mediapipe as mp
import numpy as np
import json
import time
from tensorflow.keras.models import load_model

# =============================
# Load trained model (.h5 format)
# =============================
model = load_model("model.h5")

# Load label map and sort by index
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Ensure labels are sorted properly
categories = [label for _, label in sorted(label_map.items(), key=lambda x: int(x[0]))]

# =============================
# Initialize MediaPipe
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

IMG_SIZE = 64

# =============================
# Webcam Capture
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # fallback to default camera

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Bounding box for hand
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = max(int(min(x_coords) * w) - 20, 0)
            y_min = max(int(min(y_coords) * h) - 20, 0)
            x_max = min(int(max(x_coords) * w) + 20, w)
            y_max = min(int(max(y_coords) * h) + 20, h)

            # Region of interest (ROI)
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            try:
                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = np.expand_dims(roi_normalized, axis=0)

                # Prediction
                predictions = model.predict(roi_input, verbose=0)
                class_id = np.argmax(predictions)
                confidence = np.max(predictions)

                # Display result
                if confidence > 0.5:
                    label = f"{categories[class_id]} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 2)
            except Exception as e:
                print("Skipping frame due to error:", e)

    # =============================
    # Show FPS for debugging
    # =============================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

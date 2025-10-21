import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model and label classes
model = load_model("sign_language_landmarks_model.h5")
labels = np.load("label_classes.npy", allow_pickle=True)

# Initialize MediaPipe Hands (use static_image_mode=False for real-time)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,     # For video stream
    max_num_hands=1,             # Detect only one hand
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start webcam feed
cap = cv2.VideoCapture(0)

print("✅ Press 'Q' to quit the window.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Ensure landmarks have correct length (63 = 21 points * 3 coords)
            if len(landmarks) == 63:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                pred_index = np.argmax(prediction)
                pred_label = labels[pred_index]
                confidence = np.max(prediction)

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Show prediction
                cv2.putText(
                    frame,
                    f"{pred_label} ({confidence*100:.1f}%)",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Sign Language Detection", frame)

    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

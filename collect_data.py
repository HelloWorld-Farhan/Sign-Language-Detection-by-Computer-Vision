import cv2
import os
import time
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(1)

# Setup
current_label = 'R'
max_images_per_class = 3000
base_dir = 'data'
os.makedirs(base_dir, exist_ok=True)

# function to get current count for label
def get_image_count(label):
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    return len(os.listdir(label_dir))

count = get_image_count(current_label)

print("Press a letter key (A–Z) to start collecting for that sign.")
print("Press ESC to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Save image only when hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            label_dir = os.path.join(base_dir, current_label)
            os.makedirs(label_dir, exist_ok=True)

            if count < max_images_per_class:
                img_name = os.path.join(label_dir, f"{count}.jpg")
                cv2.imwrite(img_name, image)
                count += 1

    # Show current label and count
    cv2.putText(image, f"Label: {current_label} | Count: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Sign Language Image Collector", image)

    key = cv2.waitKey(1)

    # Quit on ESC
    if key == 27:
        break

    # If A-Z is pressed, switch label
    if 65 <= key <= 90 or 97 <= key <= 122:
        current_label = chr(key).upper()
        count = get_image_count(current_label)  # ✅ resume from existing count
        print(f"\nSwitched to label: {current_label} (starting at {count})")
        time.sleep(1)  # short pause to avoid accidental clicks

cap.release()
cv2.destroyAllWindows()

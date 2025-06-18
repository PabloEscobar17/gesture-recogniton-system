import numpy as np
import cv2
import time
import pyautogui  # For taking screenshots and controlling keyboard/mouse
import webbrowser  # For opening websites
import tensorflow as tf
from mediapipe.python.solutions.hands import Hands

model = tf.keras.models.load_model("gesture_recognition_model.h5")

gesture_labels = np.load("gesture_labels.npy", allow_pickle=True)

cap = cv2.VideoCapture(0)

hands = Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)


def perform_action(gesture):
    if gesture == "call me":
        print("📸 Taking screenshot...")
        pyautogui.screenshot().save("screenshot.png")
    elif gesture == "open palm":
        print("🌐 Opening Google...")
        webbrowser.open("https://www.google.com")
    elif gesture == "close palm":
        print("🛑 Closing current window...")
        pyautogui.hotkey("alt", "f4")
    elif gesture == "thumbs up":
        print("🔊 Increasing volume...")
        pyautogui.press("volumeup")
    elif gesture == "thumbs down":
        print("🔇 Muting volume...")
        pyautogui.press("volumedown")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for point in hand_landmarks.landmark:
                landmarks.append(point.x)
                landmarks.append(point.y)

        landmarks = np.array(landmarks).reshape(1, -1)

        prediction = model.predict(landmarks)
        gesture_index = np.argmax(prediction)
        gesture_name = gesture_labels[gesture_index]

        cv2.putText(frame, f"Detected: {gesture_name}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        perform_action(gesture_name)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

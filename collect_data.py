import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "gesture_data.csv"
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]).to_csv(DATA_FILE, index=False)

cap = cv2.VideoCapture(0)
gesture_name = input("Enter gesture name: ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            df = pd.read_csv(DATA_FILE)
            df.loc[len(df)] = [gesture_name] + landmarks
            df.to_csv(DATA_FILE, index=False)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Gesture Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

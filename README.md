# 🖐️ Hand Gesture Recognition & Automation System

A Python-based real-time hand gesture recognition system using **TensorFlow**, **MediaPipe**, and **OpenCV**, capable of recognizing gestures from a webcam and mapping them to real-world actions like taking screenshots, adjusting volume, or opening websites.

## 🚀 Features

- Real-time gesture recognition from webcam
- Train your own custom gestures with label support
- Perform actions like:
  - 🖼️ Take screenshots
  - 🌐 Open websites
  - 🔊 Control system volume
  - ❌ Close windows
- Modular structure: easy to add new gestures

## 📦 Project Structure

| File                    | Description |
|-------------------------|-------------|
| `collect_data.py`       | Collects hand gesture coordinates with label input |
| `train_model.py`        | Trains and saves a neural network gesture model |
| `main.py`               | Displays real-time predictions using webcam |
| `gesture_recognition.py`| Performs system actions based on recognized gestures |
| `update_labels.py`      | Saves or updates gesture label list |

## 🛠️ Requirements

Install dependencies using:

```bash
pip install opencv-python mediapipe tensorflow pandas scikit-learn pyautogui


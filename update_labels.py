import numpy as np

gesture_labels = ["thumbs up", "open palm", "call me", "thumbs down", "close palm"]

np.save("gesture_labels.npy", gesture_labels)

print("✅ Updated gesture labels saved successfully!")

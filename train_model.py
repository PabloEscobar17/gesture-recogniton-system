import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = "gesture_data.csv"
df = pd.read_csv(DATA_FILE)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation="softmax")
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))

model.save("gesture_recognition_model.h5")

np.save("gesture_labels.npy", label_encoder.classes_)

print("✅ Model training complete and saved as 'gesture_recognition_model.h5'!")

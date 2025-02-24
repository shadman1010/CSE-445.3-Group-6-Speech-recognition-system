 
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset (Modify path accordingly)
DATASET_PATH = "path_to_audio_files"
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Prepare data
X, Y = [], []
for emotion in emotions:
    emotion_path = os.path.join(DATASET_PATH, emotion)
    for file in os.listdir(emotion_path):
        file_path = os.path.join(emotion_path, file)
        features = extract_features(file_path)
        X.append(features)
        Y.append(emotion)

X = np.array(X)
Y = np.array(Y)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model (SVM Classifier)
model = SVC(kernel="linear")
model.fit(X_train, Y_train)

# Evaluate model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

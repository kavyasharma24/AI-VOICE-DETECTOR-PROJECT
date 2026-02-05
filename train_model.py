import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

from utils.audioprocessing import extract_features

# ============================
# Dataset Folders
# ============================
DATA_DIR = {
    "dataset/human": 0,
    "dataset/ai": 1
}

# ============================
# Training Function
# ============================
def train():
    data = []

    print("Step 1: Extracting features...")

    for folder, label in DATA_DIR.items():
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        for filename in os.listdir(folder):
            if filename.endswith(".mp3"):
                path = os.path.join(folder, filename)

                try:
                    y, sr = librosa.load(path, sr=16000)
                    feat = extract_features(y, sr)

                    data.append(np.append(feat, label))

                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    if not data:
        print("❌ No audio data found!")
        return

    print("Step 2: Training model...")

    df = pd.DataFrame(data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # ✅ Save Correct Name
    joblib.dump(model, "trained_model.pkl")

    print("✅ Model trained successfully!")
    print("✅ File saved: trained_model.pkl")


if __name__ == "__main__":
    train()

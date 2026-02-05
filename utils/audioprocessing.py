import numpy as np
import librosa


# ============================
# Feature Extraction Function
# ============================
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(spectral_centroid)

    features = np.hstack([mfcc_mean, zcr_mean, centroid_mean])
    return features


# ============================
# Explanation Generator
# ============================
def get_explanation(classification):

    if classification == "AI_GENERATED":
        return "Robotic tone, unnatural pitch consistency detected."

    elif classification == "HUMAN":
        return "Natural pitch variation and realistic speech patterns detected."

    return "No explanation available."

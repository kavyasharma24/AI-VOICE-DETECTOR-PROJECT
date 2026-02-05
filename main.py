import io
import base64
import joblib
import librosa

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.audioprocessing import extract_features, get_explanation


from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# API Key (Hackathon Secret)
# ============================
API_KEY = "sk_test_123456789"

# ============================
# Supported Languages
# ============================
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ============================
# Load Model
# ============================
MODEL_PATH = "trained_model.pkl"   # ✅ Correct model file name

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except:
    model = None
    print("❌ Model Not Found! Please train first.")


# ============================
# Request Schema
# ============================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# ============================
# Home Route
# ============================
@app.get("/")
def home():
    return {"message": "AI Voice Detector API is running successfully!"}


# ============================
# Main Detection Endpoint
# ============================
@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):

    # ✅ API Key Validation
    if x_api_key is None or x_api_key != API_KEY:
        return {"status": "error", "message": "Invalid API key or missing authentication"}

    # ✅ Model Check
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Train first.")

    # ✅ Language Validation
    if request.language not in SUPPORTED_LANGUAGES:
        return {
            "status": "error",
            "message": f"Unsupported language. Only {SUPPORTED_LANGUAGES} allowed."
        }

    # ✅ Format Validation
    if request.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Invalid audio format. Only MP3 supported."}

    try:
        # ============================
        # Decode Base64 Audio
        # ============================
        audio_data = base64.b64decode(request.audioBase64)
        audio_file = io.BytesIO(audio_data)

        # Load Audio
        y, sr = librosa.load(audio_file, sr=16000)

        # Extract Features
        features = extract_features(y, sr)

        # Predict
        prediction_idx = model.predict([features])[0]
        probs = model.predict_proba([features])[0]

        classification = "HUMAN" if prediction_idx == 0 else "AI_GENERATED"
        confidence = float(max(probs))

        # ✅ Explanation Fix (Only 1 argument)
        explanation = get_explanation(classification)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

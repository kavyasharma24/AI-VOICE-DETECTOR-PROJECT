import io
import base64
import joblib
import librosa

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.audioprocessing import extract_features, get_explanation


# ============================
# App Init
# ============================
app = FastAPI(title="AI Voice Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # hackathon safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# API Key (Hackathon)
# ============================
API_KEY = "sk_test_123456789"

# ============================
# Supported Languages
# ============================
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# ============================
# Load Model
# ============================
MODEL_PATH = "trained_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    model = None
    print("❌ Model Not Found:", e)


# ============================
# Request Schema
# ============================
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# ============================
# Root Routes (Hackathon Fix)
# ============================
@app.get("/")
def home():
    return {
        "message": "AI Voice Detector API is live",
        "endpoint": "/api/voice-detection",
        "method": "POST"
    }


@app.post("/")
def root_post():
    # 👈 Hackathon POST tester fix
    return {
        "message": "POST received successfully",
        "use_endpoint": "/api/voice-detection"
    }


# ============================
# Main Detection Endpoint
# ============================
@app.post("/api/voice-detection")
async def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):

    # 🔐 API Key Validation
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    # 🤖 Model Check
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Train model first."
        )

    # 🌐 Language Validation
    if request.language not in SUPPORTED_LANGUAGES:
        return {
            "status": "error",
            "message": f"Unsupported language. Only {SUPPORTED_LANGUAGES} allowed."
        }

    # 🎵 Audio Format Check
    if request.audioFormat.lower() != "mp3":
        return {
            "status": "error",
            "message": "Invalid audio format. Only MP3 supported."
        }

    try:
        # ============================
        # Decode Base64 Audio
        # ============================
        audio_bytes = base64.b64decode(request.audioBase64)
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio
        y, sr = librosa.load(audio_buffer, sr=16000)

        # Feature extraction
        features = extract_features(y, sr)

        # Prediction
        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]

        classification = "HUMAN" if prediction == 0 else "AI_GENERATED"
        confidence = float(max(probs))

        explanation = get_explanation(classification)

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }

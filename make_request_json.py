import base64
import json
import requests

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_test_123456789"

# ============================
# Audio file path (बस filename change करो)
# ============================
AUDIO_FILE = "dataset/human/1.mp3"

# Convert MP3 → Base64
with open(AUDIO_FILE, "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode("utf-8")

# Auto JSON Request
payload = {
    "language": "Hindi",
    "audioFormat": "mp3",
    "audioBase64": audio_base64
}

headers = {
    "x-api-key": API_KEY
}

# Send Request
response = requests.post(API_URL, json=payload, headers=headers)

print("\n✅ Response:\n")
print(json.dumps(response.json(), indent=4))

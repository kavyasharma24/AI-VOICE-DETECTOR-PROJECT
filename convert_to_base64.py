import os
import base64
import json
from datetime import datetime

# -----------------------------
# Dataset Folder
# -----------------------------
DATASET_FOLDER = "dataset/human"

# -----------------------------
# List MP3 Files
# -----------------------------
files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".mp3")]

if not files:
    print("❌ No MP3 files found!")
    exit()

print("\n✅ Available MP3 Files:\n")

for i, file in enumerate(files, start=1):
    print(f"{i}. {file}")

# -----------------------------
# Select File
# -----------------------------
choice = int(input("\nEnter file number: "))

if choice < 1 or choice > len(files):
    print("❌ Invalid choice!")
    exit()

selected_file = files[choice - 1]
file_path = os.path.join(DATASET_FOLDER, selected_file)

print("\n✅ Selected:", selected_file)

# -----------------------------
# Convert to Base64
# -----------------------------
with open(file_path, "rb") as audio_file:
    base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

print("✅ Base64 conversion done!")

# -----------------------------
# Create JSON Request
# -----------------------------
request_json = {
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": base64_audio
}

# -----------------------------
# Save JSON with Timestamp Name
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"request_{timestamp}.json"

with open(output_file, "w") as f:
    json.dump(request_json, f, indent=4)

print("\n✅ JSON File Generated Successfully!")
print("📌 File Name:", output_file)
print("📌 Location:", os.getcwd())

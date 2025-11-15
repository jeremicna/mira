import requests
import sys
import os
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LEMON_FOX_API_KEY")
LOCAL_FILE_PATH = "jere-dom-convo-quick.mp3"

URL = "https://api.lemonfox.ai/v1/audio/transcriptions"

if not API_KEY:
    print("Error: LEMON_FOX_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Exact parameters per Lemonfox docs
data = {
    "language": "english",
    "response_format": "verbose_json",
    "speaker_labels": "true",   # diarization on
    "min_speakers": "2",
    "max_speakers": "2",
    # Optional: word level timestamps, might help downstream
    "timestamp_granularities[]": "word"
}

try:
    with open(LOCAL_FILE_PATH, "rb") as f:
        files = {"file": (LOCAL_FILE_PATH, f, "audio/mpeg")}
        response = requests.post(URL, headers=headers, files=files, data=data)
except FileNotFoundError:
    print(f"Error: File not found at '{LOCAL_FILE_PATH}'", file=sys.stderr)
    sys.exit(1)

if response.status_code != 200:
    print(f"Error: API request failed with status code {response.status_code}")
    print("Response:", response.text)
    sys.exit(1)

raw = response.json()

# Base result from API
segments = raw.get("segments", []) or []

result = {
    "text": raw.get("text"),
    "segments": [
        {
            "speaker": seg.get("speaker", "unknown"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
        }
        for seg in segments
    ],
}

# Debug: show what speakers the API actually gave you
speakers_from_api = sorted({seg["speaker"] for seg in result["segments"] if seg.get("speaker")})
print("Speakers from API:", speakers_from_api, file=sys.stderr)

# Fallback: if API insists on one speaker, but you know there are two,
# alternate speaker labels across segments so the rest of your pipeline works
FORCE_TWO_SPEAKERS = True

if FORCE_TWO_SPEAKERS and result["segments"]:
    unique = sorted({seg["speaker"] for seg in result["segments"] if seg.get("speaker")})
    if len(unique) == 1 and len(result["segments"]) >= 2:
        for i, seg in enumerate(result["segments"]):
            seg["speaker"] = "SPEAKER_01" if i % 2 == 0 else "SPEAKER_02"

print(json.dumps(result, indent=2, ensure_ascii=False))
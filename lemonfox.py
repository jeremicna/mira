import requests
import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("LEMON_FOX_API_KEY")
URL = "https://api.lemonfox.ai/v1/audio/transcriptions"

def get_transcription_json(file_path, min_speakers=2, max_speakers=2):
    """
    Transcribes an audio file and returns the processed JSON.
    
    Args:
        file_path (str): The path to the local audio file.
        min_speakers (int): The minimum number of speakers.
        max_speakers (int): The maximum number of speakers.

    Returns:
        dict: The processed transcription JSON, or None on failure.
    """
    if not API_KEY:
        print("Error: LEMON_FOX_API_KEY environment variable not set.", file=sys.stderr)
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "language": "english",
        "response_format": "verbose_json",
        "speaker_labels": "true",
        "min_speakers": str(min_speakers),
        "max_speakers": str(max_speakers),
        "timestamp_granularities[]": "word"
    }

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "audio/mpeg")}
            response = requests.post(URL, headers=headers, files=files, data=data)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error opening or reading file: {e}", file=sys.stderr)
        return None

    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}", file=sys.stderr)
        print(f"Response: {response.text}", file=sys.stderr)
        return None

    raw = response.json()

    segments = raw.get("segments", []) or []

    result = {
        "segments": [
            {
                "speaker": seg.get("speaker", "unknown"),
                "text": seg.get("text"),
            }
            for seg in segments
        ],
    }

    speakers_from_api = sorted({seg["speaker"] for seg in result["segments"] if seg.get("speaker")})
    print(f"Speakers from API: {speakers_from_api}", file=sys.stderr) # Print to stderr

    FORCE_TWO_SPEAKERS = True

    if FORCE_TWO_SPEAKERS and result["segments"]:
        unique = sorted({seg["speaker"] for seg in result["segments"] if seg.get("speaker")})
        if len(unique) == 1 and len(result["segments"]) >= 2:
            print("Forcing two speakers...", file=sys.stderr) # Print to stderr
            for i, seg in enumerate(result["segments"]):
                seg["speaker"] = "SPEAKER_01" if i % 2 == 0 else "SPEAKER_02"

    return result
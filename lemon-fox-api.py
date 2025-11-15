import requests
import sys
import os
from dotenv import load_dotenv  # <-- NEW LINE

load_dotenv()  # <-- NEW LINE

API_KEY = os.getenv("LEMON_FOX_API_KEY")
LOCAL_FILE_PATH = "english_convo_tst.mp3"

url = "https://api.lemonfox.ai/v1/audio/transcriptions"

if not API_KEY:
    print("Error: LEMON_FOX_API_KEY environment variable not set.", file=sys.stderr)
    # V-- NEW LINE BELOW --V
    print("Could not find LEMON_FOX_API_KEY in your .env file or environment.", file=sys.stderr)
    sys.exit(1)
    
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "language": "english",
    "response_format": "json",
    "diarization": True  # <-- NEW LINE
}

json_response = None

try:
    with open(LOCAL_FILE_PATH, "rb") as f:
        files = {"file": (LOCAL_FILE_PATH, f, "audio/mpeg")}

        # V-- PRINT MESSAGE UPDATED BELOW --V
        print(f"Uploading '{LOCAL_FILE_PATH}' for transcription (with diarization)...")
        
        response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 200:
            print("Transcription successful.")
            
            json_response = response.json()
            
            print("\n--- API Response ---")
            print(json_response)
            print("--------------------")
            
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print("Response:", response.text)

except FileNotFoundError:
    print(f"Error: File not found at '{LOCAL_FILE_PATH}'", file=sys.stderr)
    print("If your script is not in the 'MIRA' directory, you may need to provide the full path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)

if json_response:
    print("\nScript finished. 'json_response' variable contains the result.")
else:
    print("\nScript finished, but 'json_response' is empty due to an error.")
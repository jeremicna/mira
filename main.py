import uuid
import time
import threading
from flask import Flask, request, jsonify
from audiotap import AudioTap
from lemonfox import get_transcription_json
from db import lmdb_get_json, lmdb_put_json
from gemini import get_json_analysis


app = Flask(__name__)

audio = AudioTap(
    url="rtsp://localhost:8554/mystream",
    sample_rate=48000,
    channels=1,
    buffer_seconds=180 # keep 3 minutes of audio
)
audio.start()

CURRENT_FACE = None
FACE_START_TIME = None
LOCK = threading.Lock()


@app.post("/api/face")
def face_update():
    """
    Body format:
       { "uuid": "<uuid>" }
    or
       { "uuid": null }
    """
    global CURRENT_FACE, FACE_START_TIME

    data = request.get_json()
    face_id = data.get("uuid", None)

    # Convert empty string to None
    if face_id in ("", None):
        new_face = None
    else:
        # Validate uuid format
        try:
            new_face = str(uuid.UUID(face_id))
        except Exception:
            return jsonify({"error": "invalid uuid"}), 400

    with LOCK:
        # ---------------------------------------------------
        # CASE 1: Face appears
        # ---------------------------------------------------
        if new_face and CURRENT_FACE is None:
            CURRENT_FACE = new_face
            FACE_START_TIME = time.time()
            print(f"[FACE APPEARED] {CURRENT_FACE} — starting recording")
            return jsonify({"active_face": CURRENT_FACE})

        # ---------------------------------------------------
        # CASE 2: Face disappears
        # ---------------------------------------------------
        if new_face is None and CURRENT_FACE is not None:
            face = CURRENT_FACE
            start_ts = FACE_START_TIME
            end_ts = time.time()

            CURRENT_FACE = None
            FACE_START_TIME = None

            print(f"[FACE DISAPPEARED] {face} — saving audio")

            # process asynchronously
            threading.Thread(
                target=process_audio_capture,
                args=(start_ts, end_ts, face),
                daemon=True
            ).start()

            return jsonify({"active_face": None})

        # ---------------------------------------------------
        # CASE 3: Same face still present → nothing changes
        # ---------------------------------------------------
        return jsonify({"active_face": CURRENT_FACE})


# -------------------------------------------
# BACKGROUND AUDIO PROCESSING
# -------------------------------------------
def process_audio_capture(start_ts, end_ts, face_id):
    duration = end_ts - start_ts
    pcm = audio.get_last(duration)

    if pcm is None:
        print("WARN: Not enough audio in buffer.")
        return

    mp3_path = f"capture_{face_id}_{int(start_ts)}.mp3"
    audio.save_mp3(pcm, mp3_path)
    print(f"Saved MP3: {mp3_path}")

    result = get_transcription_json(mp3_path)
    if result:
        print(f"Transcription for {face_id}: {result}")
    else:
        print("Transcription failed")

    trans_key = f"transcriptions:{face_id}"

    existing_list = lmdb_get_json(trans_key)
    if existing_list is None:
        existing_list = []

    existing_list.append(result)

    lmdb_put_json(trans_key, existing_list)
    print(f"LMDB updated: {trans_key} now has {len(existing_list)} chunks")

    analysis_json = get_json_analysis(existing_list)

    analysis_key = f"gemini:{face_id}"
    lmdb_put_json(analysis_key, analysis_json)

    print(f"Analysis stored: {analysis_key}")


if __name__ == "__main__":
    print("Starting audio tap…")
    app.run(host="0.0.0.0", port=8000)

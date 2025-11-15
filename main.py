import uuid
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.post("/api/face")
def link_face():
    data = request.get_json()
    face_id = data.get("uuid")

    try:
        face_uuid = uuid.UUID(face_id)
    except Exception:
        return jsonify({"error": "invalid uuid"}), 400

    return jsonify({"linked_face": str(face_uuid)})
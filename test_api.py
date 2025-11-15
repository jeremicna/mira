import threading
import time
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

LOCK = threading.Lock()

# ------------------------------------------------------------------
# States
# ------------------------------------------------------------------

STATE_EMPTY = {
    "active_face": None,
    "gemini": None,
    "has_gemini": False
}

STATE_FULL = {
    "active_face": True,
    "gemini": {
        "name": "Alice Clark",
        "occupation": "Teacher",
        "relationship": "Daughter",
        "current_state": (
            "Alice said work has been stressful lately and she's preparing "
            "for the school fundraiser next week."
        ),
        "last_points": [
            "You talked about your doctor visit on Monday",
            "Alice explained the new medication schedule",
            "You asked about her kids and Jake scored a goal"
        ],
        "convo_points": [
            "Ask how her kids are doing this week",
            "Ask if Sunday still works for the call"
        ]
    },
    "has_gemini": True
}


CURRENT = STATE_EMPTY.copy()


# ==================================================================
#  ROUTES
# ==================================================================
@app.get("/api/state")
def api_state():
    with LOCK:
        return jsonify(CURRENT)


# ==================================================================
# AUTO-TOGGLER THREAD
# ==================================================================
def auto_toggle():
    global CURRENT
    state = False  # False = empty, True = full

    while True:
        with LOCK:
            if state:
                CURRENT = STATE_FULL.copy()
                print("→ Switched to FULL")
            else:
                CURRENT = STATE_EMPTY.copy()
                print("→ Switched to EMPTY")

        state = not state
        time.sleep(10)  # toggle every 10 seconds


# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    threading.Thread(target=auto_toggle, daemon=True).start()

    print("Running auto-toggle Flask API on http://localhost:8000/api/state …")
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)

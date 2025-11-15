import cv2
import numpy as np
import uuid
import requests
import json
import os
from collections import deque, Counter
import time

# Configuration
RTSP_URL = "rtsp://localhost:8554/mystream"
API_URL = "http://localhost:5000/api/face"
KNOWN_FACES_FILE = "known_faces.json"
SIMILARITY_THRESHOLD = 0.9  # Higher = stricter matching (cosine similarity)
FRAME_SKIP = 2   # Process every Nth frame for performance
BUFFER_DURATION = 3.0  # seconds - how long to track before confirming face change
FPS_ESTIMATE = 30  # Estimated FPS for buffer size calculation

class FaceTracker:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.current_face_uuid = None

        buffer_size = int(FPS_ESTIMATE * BUFFER_DURATION / FRAME_SKIP)
        self.frame_buffer = deque(maxlen=buffer_size)

        self.last_face_change_time = time.time()

        self.last_unknown_face = None  # store image for possible registration

        # Load OpenCV DNN face detector
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"

        self.download_models(model_path, weights_path)
        self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)

        self.load_known_faces()

        print(f"Face tracker initialized with {buffer_size}-frame buffer (~{BUFFER_DURATION}s)")

    def download_models(self, model_path, weights_path):
        if not os.path.exists(model_path):
            print("Downloading face detection model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                model_path
            )

        if not os.path.exists(weights_path):
            print("Downloading face detection weights (this may take a minute)...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                weights_path
            )

    def load_known_faces(self):
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, 'r') as f:
                data = json.load(f)
                self.known_face_encodings = [np.array(enc) for enc in data['encodings']]
                self.known_face_ids = data['ids']
            print(f"Loaded {len(self.known_face_ids)} known faces")

    def save_known_faces(self):
        with open(KNOWN_FACES_FILE, 'w') as f:
            json.dump({
                'encodings': [enc.tolist() for enc in self.known_face_encodings],
                'ids': self.known_face_ids
            }, f, indent=2)

    def compute_face_histogram(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def compare_faces(self, hist1, hist2):
        return np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))

    # ----------------------------------------------------------
    # NEW: identify ONLY, no new UUID creation
    # ----------------------------------------------------------
    def identify_face(self, face_img):
        """Returns known UUID or 'UNKNOWN' (no registration here)."""
        face_hist = self.compute_face_histogram(face_img)

        if len(self.known_face_encodings) == 0:
            self.last_unknown_face = face_img
            return "UNKNOWN"

        best_match_idx = -1
        best_similarity = 0

        for idx, known_hist in enumerate(self.known_face_encodings):
            similarity = self.compare_faces(face_hist, known_hist)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx

        if best_similarity >= SIMILARITY_THRESHOLD:
            return self.known_face_ids[best_match_idx]

        self.last_unknown_face = face_img
        return "UNKNOWN"

    # ----------------------------------------------------------
    # NEW: register a new face only after consensus
    # ----------------------------------------------------------
    def register_new_face(self, face_img):
        face_hist = self.compute_face_histogram(face_img)
        face_uuid = str(uuid.uuid4())
        self.known_face_encodings.append(face_hist)
        self.known_face_ids.append(face_uuid)
        self.save_known_faces()
        print(f"[REGISTERED NEW FACE] UUID: {face_uuid}")
        return face_uuid

    def post_face_change(self, face_uuid):
        try:
            response = requests.post(
                API_URL,
                json={"uuid": face_uuid},
                timeout=2
            )
            print(f"API Response: {response.json()}")
        except Exception as e:
            print(f"Error posting to API: {e}")

    def get_stable_face(self):
        """Return stable UUID only if >50% consensus."""
        if len(self.frame_buffer) == 0:
            return None

        face_counts = Counter(self.frame_buffer)
        most_common_face, count = face_counts.most_common(1)[0]

        confidence = count / len(self.frame_buffer)

        if confidence > 0.5:
            return most_common_face

        return self.current_face_uuid

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.detector.setInput(blob)
        detections = self.detector.forward()

        detected_uuid = None
        face_locations = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                face_locations.append((startX, startY, endX, endY))

                face_img = frame[startY:endY, startX:endX]

                if face_img.size > 0:
                    detected_uuid = self.identify_face(face_img)
                    break

        # Add detection to buffer
        self.frame_buffer.append(detected_uuid)

        # ----------------------------------------------------------
        # CONSENSUS: decide whether UNKNOWN becomes a new UUID
        # ----------------------------------------------------------
        if detected_uuid == "UNKNOWN":
            unknown_count = self.frame_buffer.count("UNKNOWN")
            required_frames = int(self.frame_buffer.maxlen * 0.8)

            if len(self.frame_buffer) >= required_frames:
                confidence = unknown_count / len(self.frame_buffer)

                if confidence > 0.6 and self.last_unknown_face is not None:
                    print(f"[CONSENSUS] UNKNOWN confirmed ({unknown_count}/{len(self.frame_buffer)})")
                    detected_uuid = self.register_new_face(self.last_unknown_face)

                    # Reset buffer so new ID gets tracked immediately
                    self.frame_buffer.clear()
                    self.frame_buffer.append(detected_uuid)

        # Get stable consensus UUID
        stable_uuid = self.get_stable_face()

        current_time = time.time()

        if stable_uuid != self.current_face_uuid:
            if len(self.frame_buffer) >= self.frame_buffer.maxlen * 0.8:
                print(f"Face change: {self.current_face_uuid} -> {stable_uuid}")
                self.current_face_uuid = stable_uuid
                self.last_face_change_time = current_time
                self.post_face_change(stable_uuid)

        return face_locations


def main():
    print("Connecting to MediaMTX stream...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("Error: Could not connect to stream")
        return

    print("Stream connected. Starting face detection...")
    tracker = FaceTracker()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                continue

            tracker.process_frame(frame)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        if tracker.current_face_uuid:
            tracker.post_face_change(None)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

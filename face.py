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
API_URL = "http://localhost:8000/api/face"
KNOWN_FACES_FILE = "known_faces.json"
CONFIDENCE_THRESHOLD = 90
FRAME_SKIP = 2
BUFFER_DURATION = 5.0
FPS_ESTIMATE = 30
FACE_CHANGE_CONFIRMATION_TIME = 2.0


class FaceTracker:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        self.known_face_ids = []
        self.known_face_images = []
        self.trained = False
        self.current_face_uuid = None

        buffer_size = int(FPS_ESTIMATE * BUFFER_DURATION / FRAME_SKIP)
        self.frame_buffer = deque(maxlen=buffer_size)

        self.last_face_change_time = time.time()
        self.pending_face_uuid = None
        self.pending_face_start_time = None

        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"

        self.download_models(model_path, weights_path)
        self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)

        self.load_known_faces()

        print(f"Face tracker initialized with {buffer_size}-frame buffer (~{BUFFER_DURATION}s)")
        print(f"LBPH Recognizer ready. Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Face change confirmation time: {FACE_CHANGE_CONFIRMATION_TIME}s")
        print("MODE: Match only - no new face registration")

    # -----------------------------------------------------------
    # Download Models if Missing
    # -----------------------------------------------------------

    def download_models(self, model_path, weights_path):
        if not os.path.exists(model_path):
            print("Downloading face detection model...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                model_path
            )

        if not os.path.exists(weights_path):
            print("Downloading face detection weights...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                weights_path
            )

    # -----------------------------------------------------------
    # Load Known Faces
    # -----------------------------------------------------------

    def load_known_faces(self):
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, 'r') as f:
                data = json.load(f)

            if 'faces' in data and len(data['faces']) > 0:
                self.known_face_ids = []
                self.known_face_images = []

                for face_data in data['faces']:
                    face_id = face_data['id']
                    for img_hex in face_data['images']:
                        img_bytes = np.frombuffer(bytes.fromhex(img_hex), dtype=np.uint8)
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

                        self.known_face_ids.append(face_id)
                        self.known_face_images.append(img)

                if len(self.known_face_images) > 0:
                    label_mapping = {id_: idx for idx, id_ in enumerate(set(self.known_face_ids))}
                    labels = [label_mapping[id_] for id_ in self.known_face_ids]

                    self.recognizer.train(self.known_face_images, np.array(labels))
                    self.trained = True
                    self.label_to_uuid = {v: k for k, v in label_mapping.items()}

                    print(f"Loaded {len(set(self.known_face_ids))} known people.")
                else:
                    print("No valid images found.")
            else:
                print("No known faces in JSON.")
        else:
            print("WARNING: No known_faces.json found.")

    # -----------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------

    def preprocess_face(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128))
        gray = cv2.equalizeHist(gray)
        return gray

    # -----------------------------------------------------------
    # Recognition
    # -----------------------------------------------------------

    def identify_face(self, face_img):
        if not self.trained:
            return "UNKNOWN"

        gray = self.preprocess_face(face_img)

        try:
            label, confidence = self.recognizer.predict(gray)

            if confidence < CONFIDENCE_THRESHOLD:
                face_uuid = self.label_to_uuid[label]
                print(f"Recognized: {face_uuid} (confidence: {confidence:.1f})")
                return face_uuid
            else:
                print(f"Unknown face (confidence {confidence:.1f})")
                return "UNKNOWN"

        except Exception as e:
            print(f"Error during recognition: {e}")
            return "UNKNOWN"

    # -----------------------------------------------------------
    # Post event to API
    # -----------------------------------------------------------

    def post_face_change(self, face_uuid):
        try:
            response = requests.post(
                API_URL,
                json={"uuid": face_uuid},
                timeout=2
            )
            print(f"[API] POST → {face_uuid}: {response.json()}")
        except Exception as e:
            print(f"[API ERROR] {e}")

    # -----------------------------------------------------------
    # Decide stable face
    # -----------------------------------------------------------

    def get_stable_face(self):
        if len(self.frame_buffer) == 0:
            return None

        face_counts = Counter(self.frame_buffer)
        most_common, count = face_counts.most_common(1)[0]

        if (count / len(self.frame_buffer)) > 0.5:
            return most_common

        return self.current_face_uuid

    # -----------------------------------------------------------
    # Main frame processing
    # -----------------------------------------------------------

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.detector.setInput(blob)
        detections = self.detector.forward()

        detected_uuid = None

        # Detect 1 face max
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_img = frame[y1:y2, x1:x2]

                if face_img.size > 0:
                    detected_uuid = self.identify_face(face_img)

                break

        # Add to rolling buffer
        self.frame_buffer.append(detected_uuid)

        stable_uuid = self.get_stable_face()
        now = time.time()

        # -------------------------------------------------------
        # FACE CHANGE LOGIC WITH FULL SAFETY FIX
        # -------------------------------------------------------

        if stable_uuid != self.current_face_uuid:

            # New candidate face
            if stable_uuid != self.pending_face_uuid:
                self.pending_face_uuid = stable_uuid
                self.pending_face_start_time = now
                print(f"[PENDING] {stable_uuid} (waiting {FACE_CHANGE_CONFIRMATION_TIME}s)")

            else:
                # Must check timer safely
                if self.pending_face_start_time is not None:
                    time_elapsed = now - self.pending_face_start_time

                    if time_elapsed >= FACE_CHANGE_CONFIRMATION_TIME:
                        print(f"[CONFIRMED] {self.current_face_uuid} → {stable_uuid}")

                        # ALWAYS POST (including UNKNOWN)
                        self.post_face_change(stable_uuid)

                        self.current_face_uuid = stable_uuid
                        self.last_face_change_time = now

                        # Reset pending status
                        self.pending_face_uuid = None
                        self.pending_face_start_time = None
                else:
                    # Timer missing—repair state cleanly
                    print(f"[FIX] Missing timer, restarting timer for {stable_uuid}")
                    self.pending_face_start_time = now

        else:
            # Face unchanged — clear pending
            if self.pending_face_uuid is not None:
                print("[CANCELLED] Pending change cancelled")
                self.pending_face_uuid = None
                self.pending_face_start_time = None

        return []


# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------

def main():
    print("Connecting to stream...")
    cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("ERROR: Could not connect to stream.")
        return

    tracker = FaceTracker()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed — reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(RTSP_URL)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            tracker.process_frame(frame)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        tracker.post_face_change(None)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

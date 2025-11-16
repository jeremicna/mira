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
CONFIDENCE_THRESHOLD = 90  # Lower = stricter matching for LBPH (typical range: 50-100)
FRAME_SKIP = 2   # Process every Nth frame for performance
BUFFER_DURATION = 5.0  # seconds - how long to track before confirming face change
FPS_ESTIMATE = 30  # Estimated FPS for buffer size calculation
FACE_CHANGE_CONFIRMATION_TIME = 2.0  # seconds - how long new face must be stable before confirming

class FaceTracker:
    def __init__(self):
        # OpenCV LBPH Face Recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        
        self.known_face_ids = []
        self.known_face_images = []  # Store sample images for retraining
        self.trained = False
        self.current_face_uuid = None

        buffer_size = int(FPS_ESTIMATE * BUFFER_DURATION / FRAME_SKIP)
        self.frame_buffer = deque(maxlen=buffer_size)

        self.last_face_change_time = time.time()
        self.last_unknown_face = None  # store image for possible registration
        
        # New: tracking for face change confirmation
        self.pending_face_uuid = None
        self.pending_face_start_time = None

        # Load OpenCV DNN face detector
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"

        self.download_models(model_path, weights_path)
        self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)

        self.load_known_faces()

        print(f"Face tracker initialized with {buffer_size}-frame buffer (~{BUFFER_DURATION}s)")
        print(f"LBPH Recognizer ready. Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print(f"Face change confirmation time: {FACE_CHANGE_CONFIRMATION_TIME}s")

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
        """Load known faces from JSON and retrain recognizer"""
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, 'r') as f:
                data = json.load(f)
                
            if 'faces' in data and len(data['faces']) > 0:
                self.known_face_ids = []
                self.known_face_images = []
                
                for face_data in data['faces']:
                    face_id = face_data['id']
                    # Decode base64 images back to numpy arrays
                    for img_base64 in face_data['images']:
                        img_bytes = np.frombuffer(
                            bytes.fromhex(img_base64),
                            dtype=np.uint8
                        )
                        img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)
                        
                        self.known_face_ids.append(face_id)
                        self.known_face_images.append(img)
                
                # Train the recognizer
                if len(self.known_face_images) > 0:
                    label_mapping = {id_: idx for idx, id_ in enumerate(set(self.known_face_ids))}
                    labels = [label_mapping[id_] for id_ in self.known_face_ids]
                    
                    self.recognizer.train(self.known_face_images, np.array(labels))
                    self.trained = True
                    self.label_to_uuid = {v: k for k, v in label_mapping.items()}
                    
                    print(f"Loaded {len(set(self.known_face_ids))} known faces with {len(self.known_face_images)} training images")
                else:
                    print("No valid face images found in JSON")
            else:
                print("No known faces in database")
        else:
            print("No known faces database found - will create new one")

    def save_known_faces(self):
        """Save known faces with their training images"""
        # Group images by UUID
        face_dict = {}
        for face_id, img in zip(self.known_face_ids, self.known_face_images):
            if face_id not in face_dict:
                face_dict[face_id] = []
            
            # Encode image to hex string for JSON storage
            _, img_encoded = cv2.imencode('.jpg', img)
            img_hex = img_encoded.tobytes().hex()
            face_dict[face_id].append(img_hex)
        
        # Create faces list
        faces_list = [
            {'id': face_id, 'images': images}
            for face_id, images in face_dict.items()
        ]
        
        with open(KNOWN_FACES_FILE, 'w') as f:
            json.dump({'faces': faces_list}, f, indent=2)
        
        print(f"Saved {len(face_dict)} faces to {KNOWN_FACES_FILE}")

    def preprocess_face(self, face_img):
        """Preprocess face for recognition"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        # Resize to consistent size
        gray = cv2.resize(gray, (128, 128))
        # Histogram equalization for better lighting invariance
        gray = cv2.equalizeHist(gray)
        return gray

    def identify_face(self, face_img):
        """Returns known UUID or 'UNKNOWN' (no registration here)."""
        if not self.trained:
            self.last_unknown_face = face_img
            return "UNKNOWN"
        
        gray = self.preprocess_face(face_img)
        
        try:
            label, confidence = self.recognizer.predict(gray)
            
            # LBPH: Lower confidence = better match
            # Typical good matches are < 50, anything > 100 is usually different person
            if confidence < CONFIDENCE_THRESHOLD:
                face_uuid = self.label_to_uuid[label]
                print(f"Recognized face: {face_uuid} (confidence: {confidence:.1f})")
                return face_uuid
            else:
                print(f"Unknown face (confidence: {confidence:.1f} > threshold {CONFIDENCE_THRESHOLD})")
                self.last_unknown_face = face_img
                return "UNKNOWN"
                
        except Exception as e:
            print(f"Recognition error: {e}")
            self.last_unknown_face = face_img
            return "UNKNOWN"

    def register_new_face(self, face_img):
        """Register a new face with multiple samples"""
        face_uuid = str(uuid.uuid4())
        gray = self.preprocess_face(face_img)
        
        # Add the new face
        self.known_face_ids.append(face_uuid)
        self.known_face_images.append(gray)
        
        # Retrain the recognizer with all faces
        label_mapping = {id_: idx for idx, id_ in enumerate(set(self.known_face_ids))}
        labels = [label_mapping[id_] for id_ in self.known_face_ids]
        
        self.recognizer.train(self.known_face_images, np.array(labels))
        self.trained = True
        self.label_to_uuid = {v: k for k, v in label_mapping.items()}
        
        self.save_known_faces()
        print(f"[REGISTERED NEW FACE] UUID: {face_uuid}")
        return face_uuid

    def add_training_sample(self, face_uuid, face_img):
        """Add additional training sample for existing face"""
        gray = self.preprocess_face(face_img)
        
        self.known_face_ids.append(face_uuid)
        self.known_face_images.append(gray)
        
        # Retrain
        label_mapping = {id_: idx for idx, id_ in enumerate(set(self.known_face_ids))}
        labels = [label_mapping[id_] for id_ in self.known_face_ids]
        
        self.recognizer.update([gray], np.array([label_mapping[face_uuid]]))
        
        # Save periodically (every 5 samples)
        if len(self.known_face_images) % 5 == 0:
            self.save_known_faces()
            print(f"Added training sample for {face_uuid}")

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
                    
                    # Add training samples for known faces to improve recognition
                    if detected_uuid != "UNKNOWN" and detected_uuid is not None:
                        # Randomly collect additional samples (1 in 30 frames)
                        if np.random.random() < 0.03:
                            self.add_training_sample(detected_uuid, face_img)
                    
                    break

        # Add detection to buffer
        self.frame_buffer.append(detected_uuid)

        # CONSENSUS: decide whether UNKNOWN becomes a new UUID
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

        # NEW: Handle face change with 4-second confirmation
        if stable_uuid != self.current_face_uuid:
            # If this is a new pending face, start tracking it
            if stable_uuid != self.pending_face_uuid:
                self.pending_face_uuid = stable_uuid
                self.pending_face_start_time = current_time
                print(f"[PENDING] New face detected: {stable_uuid}, waiting {FACE_CHANGE_CONFIRMATION_TIME}s for confirmation...")
            
            # Check if the pending face has been stable for 4 seconds
            elif self.pending_face_start_time is not None:
                time_elapsed = current_time - self.pending_face_start_time
                
                if time_elapsed >= FACE_CHANGE_CONFIRMATION_TIME:
                    print(f"[CONFIRMED] Face change after {time_elapsed:.1f}s: {self.current_face_uuid} -> {stable_uuid}")
                    self.current_face_uuid = stable_uuid
                    self.last_face_change_time = current_time
                    self.post_face_change(stable_uuid)
                    
                    # Reset pending tracking
                    self.pending_face_uuid = None
                    self.pending_face_start_time = None
                else:
                    print(f"[WAITING] Face {stable_uuid} stable for {time_elapsed:.1f}/{FACE_CHANGE_CONFIRMATION_TIME}s")
        else:
            # If we're back to the current face, cancel any pending change
            if self.pending_face_uuid is not None:
                print(f"[CANCELLED] Pending face change cancelled, back to {self.current_face_uuid}")
                self.pending_face_uuid = None
                self.pending_face_start_time = None

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
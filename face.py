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
SIMILARITY_THRESHOLD = 0.85  # Higher = stricter matching (cosine similarity)
FRAME_SKIP = 2   # Process every Nth frame for performance
BUFFER_DURATION = 2.0  # seconds - how long to track before confirming face change
FPS_ESTIMATE = 30  # Estimated FPS for buffer size calculation

class FaceTracker:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.current_face_uuid = None
        
        # Calculate buffer size based on FPS and desired duration
        buffer_size = int(FPS_ESTIMATE * BUFFER_DURATION / FRAME_SKIP)
        self.frame_buffer = deque(maxlen=buffer_size)
        
        self.last_face_change_time = time.time()
        
        # Load OpenCV's DNN face detector
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download models if not present
        self.download_models(model_path, weights_path)
        
        self.detector = cv2.dnn.readNetFromCaffe(model_path, weights_path)
        
        self.load_known_faces()
        
        print(f"Face tracker initialized with {buffer_size}-frame buffer (~{BUFFER_DURATION}s)")
    
    def download_models(self, model_path, weights_path):
        """Download face detection models if needed"""
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
        """Load previously seen faces from disk"""
        if os.path.exists(KNOWN_FACES_FILE):
            with open(KNOWN_FACES_FILE, 'r') as f:
                data = json.load(f)
                # Convert lists back to numpy arrays
                self.known_face_encodings = [np.array(enc) for enc in data['encodings']]
                self.known_face_ids = data['ids']
            print(f"Loaded {len(self.known_face_ids)} known faces")
    
    def save_known_faces(self):
        """Save known faces to disk"""
        with open(KNOWN_FACES_FILE, 'w') as f:
            json.dump({
                'encodings': [enc.tolist() for enc in self.known_face_encodings],
                'ids': self.known_face_ids
            }, f, indent=2)
    
    def compute_face_histogram(self, face_img):
        """Compute simple histogram feature for face matching"""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def compare_faces(self, hist1, hist2):
        """Compare two face histograms using cosine similarity"""
        similarity = np.dot(hist1, hist2) / (np.linalg.norm(hist1) * np.linalg.norm(hist2))
        return similarity
    
    def identify_face(self, face_img):
        """Match face to known faces or create new UUID"""
        face_hist = self.compute_face_histogram(face_img)
        
        if len(self.known_face_encodings) == 0:
            # First face ever seen
            face_uuid = str(uuid.uuid4())
            self.known_face_encodings.append(face_hist)
            self.known_face_ids.append(face_uuid)
            self.save_known_faces()
            print(f"New face registered: {face_uuid}")
            return face_uuid
        
        # Compare with known faces
        best_match_idx = -1
        best_similarity = 0
        
        for idx, known_hist in enumerate(self.known_face_encodings):
            similarity = self.compare_faces(face_hist, known_hist)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = idx
        
        if best_similarity >= SIMILARITY_THRESHOLD:
            # Found a match
            face_uuid = self.known_face_ids[best_match_idx]
            return face_uuid
        else:
            # New face
            face_uuid = str(uuid.uuid4())
            self.known_face_encodings.append(face_hist)
            self.known_face_ids.append(face_uuid)
            self.save_known_faces()
            print(f"New face registered: {face_uuid} (best similarity: {best_similarity:.3f})")
            return face_uuid
    
    def post_face_change(self, face_uuid):
        """Send face change to API"""
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
        """
        Get the most common face UUID from the buffer.
        Requires majority consensus (>50%) to confirm face change.
        """
        if len(self.frame_buffer) == 0:
            return None
        
        # Count occurrences of each UUID
        face_counts = Counter(self.frame_buffer)
        
        # Get most common face
        most_common_face, count = face_counts.most_common(1)[0]
        
        # Require majority (>50%) for confirmation
        buffer_size = len(self.frame_buffer)
        confidence = count / buffer_size
        
        # Only return if we have majority consensus
        if confidence > 0.5:
            return most_common_face
        
        # If no consensus, return current face to avoid flickering
        return self.current_face_uuid
    
    def process_frame(self, frame):
        """Detect and identify faces in frame"""
        h, w = frame.shape[:2]
        
        # Prepare image for face detection
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
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure box is within frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                face_locations.append((startX, startY, endX, endY))
                
                # Extract face for identification
                face_img = frame[startY:endY, startX:endX]
                
                if face_img.size > 0:
                    detected_uuid = self.identify_face(face_img)
                    break  # Use first (most confident) face
        
        # Add to buffer for temporal stability
        self.frame_buffer.append(detected_uuid)
        
        # Get stable face based on buffer consensus
        stable_uuid = self.get_stable_face()
        
        # Check if face changed with temporal confirmation
        current_time = time.time()
        if stable_uuid != self.current_face_uuid:
            # Only trigger change if buffer is full (we've seen enough frames)
            if len(self.frame_buffer) >= self.frame_buffer.maxlen * 0.8:
                time_since_last_change = current_time - self.last_face_change_time
                
                print(f"Face change detected: {self.current_face_uuid} -> {stable_uuid}")
                print(f"  Buffer consensus: {self.frame_buffer.count(stable_uuid)}/{len(self.frame_buffer)} frames")
                
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
            
            # Skip frames for performance
            if frame_count % FRAME_SKIP != 0:
                continue
            
            # Process frame
            face_locations = tracker.process_frame(frame)
            
            # Optional: Display video with face boxes
            # Uncomment to show video window
            """
            for (startX, startY, endX, endY) in face_locations:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                # Display current face UUID
                if tracker.current_face_uuid:
                    label = tracker.current_face_uuid[:8]
                    cv2.putText(frame, label, (startX, startY - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            """
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        # Send null face when shutting down
        if tracker.current_face_uuid:
            tracker.post_face_change(None)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# mira

## Overview
Mira is a real-time facial recognition based conversational intelligence engine.

Devpost: [https://devpost.com/software/mira-72v46f](https://devpost.com/software/mira-72v46f)

## Features
- OpenCV based real-time face recognition and state tracking
- Context storage via LMDB
- Audio analysis pipeline (Whisper/Lemonfox)
- Flask API exposing the current state  
- Frontend liquid-glass UI overlay  

## Tech Stack
**Backend:** Python, Flask, OpenCV, LMDB, FFmpeg/MediaMTX  
**Frontend:** HTML/CSS/JavaScript  
**Hardware:** For demo: Macbook Air Webcam, Hardware mockup: Raspberry Pi Zero 2 W + CSI camera

## Usage
Download dependencies

```pip install -r requirements.txt```

```brew install mediamtx```

Start mediamtx server ```$ mediamtx```

Stream rtsp av stream to mediamtx using ffmpeg/OBS Studio/other

Run main
```python main.py```

Run facial recognition worker
```python face.py```

Navigate to ```frontend/main.html```

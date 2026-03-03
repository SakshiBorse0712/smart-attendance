
import urllib.request
import os
from pathlib import Path

# MediaPipe Face Landmarker model URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_NAME = "face_landmarker.task"

# Determine the target directory (app/ml/)
SCRIPT_DIR = Path(__file__).parent
ML_DIR = SCRIPT_DIR / "app" / "ml"
TARGET_PATH = ML_DIR / MODEL_NAME

def download_model():
    print(f"Checking for {MODEL_NAME}...")
    ML_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_PATH.exists() and TARGET_PATH.stat().st_size > 0:
        print(f"✓ Model already exists at {TARGET_PATH}")
        return

    print(f"Downloading from: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, TARGET_PATH)
        print(f"✓ Successfully downloaded model to {TARGET_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

if __name__ == "__main__":
    download_model()

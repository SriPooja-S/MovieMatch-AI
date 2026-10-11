import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from app.utils.logger import logger
from app.core.model_manager import model_manager


class FaceProcessor:
    """
    Lightweight CPU face detector and feature extractor.
    Uses OpenCV DNN / Cascade for face detection and extracts face crops
    for actor identification and candidate filtering.
    """

    def __init__(self):
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """
        Detect face bounding boxes from an image.
        Returns list of dicts with 'box': [x, y, w, h] and 'crop': PIL Image.
        """
        try:
            img_np = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40)
            )

            results = []
            for (x, y, w, h) in faces:
                # Add padding
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_np.shape[1], x + w + pad_x)
                y2 = min(img_np.shape[0], y + h + pad_y)

                crop_np = img_np[y1:y2, x1:x2]
                crop_pil = Image.fromarray(crop_np)

                results.append({
                    "box": [int(x), int(y), int(w), int(h)],
                    "crop": crop_pil
                })

            logger.info(f"[FaceProcessor] Detected {len(results)} face(s) in image")
            return results
        except Exception as e:
            logger.error(f"[FaceProcessor] Face detection failed: {e}")
            return []

    def detect_faces_in_frames(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Extract unique face crops across multiple video frames."""
        face_crops = []
        for frame in frames:
            faces = self.detect_faces(frame)
            for f in faces:
                face_crops.append(f["crop"])
        return face_crops

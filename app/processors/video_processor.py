import numpy as np
from PIL import Image
from typing import List, Optional
from pathlib import Path
from app.utils.logger import logger


class VideoProcessor:
    """Extract frames from local video files for visual analysis."""

    SUPPORTED = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """Extract uniformly-spaced frames from a video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release()
                return []

            indices = np.linspace(0, total - 1, num_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames
        except Exception as e:
            logger.error(f"Failed to extract frames: {e}")
            return []

    def get_video_info(self, video_path: str) -> dict:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {
                "fps": fps,
                "frame_count": total,
                "width": w,
                "height": h,
                "duration": total / fps if fps > 0 else 0,
            }
        except Exception as e:
            logger.error(f"get_video_info failed: {e}")
            return {}

import io
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List, Union, Optional
from app.utils.logger import logger


class ImageProcessor:
    """Process uploaded images — OCR, color extraction, validation."""

    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def load_image(self, source: Union[str, bytes, io.BytesIO]) -> Image.Image:
        try:
            if isinstance(source, str):
                return Image.open(source).convert("RGB")
            elif isinstance(source, bytes):
                return Image.open(io.BytesIO(source)).convert("RGB")
            else:
                return Image.open(source).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise

    def extract_text(self, image: Image.Image) -> str:
        """OCR via Tesseract (graceful fallback if not installed)."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image).strip()
            logger.info(f"OCR extracted {len(text)} chars")
            return text
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def preprocess_for_search(self, image: Image.Image) -> Dict:
        """Extract all useful features from an image."""
        try:
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            ocr_text = self.extract_text(image)
            return {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.width / image.height,
                "text": ocr_text,
                "is_portrait": image.width < image.height,
            }
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return {}

    def enhance(self, image: Image.Image) -> Image.Image:
        try:
            image = ImageEnhance.Contrast(image).enhance(1.2)
            image = ImageEnhance.Sharpness(image).enhance(1.1)
            return image
        except Exception:
            return image

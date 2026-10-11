import os
import tempfile
import subprocess
from typing import Dict, Optional, Any
from app.utils.logger import logger
from app.core.model_manager import model_manager


class AudioProcessor:
    """
    CPU-optimized Speech-to-Text (ASR) audio processing engine.
    Uses CTranslate2-backed faster-whisper (tiny.en, int8 quantized) for ultra-fast,
    low-RAM (<150MB) dialogue extraction from video clips.
    """

    def __init__(self, model_size: str = "tiny.en"):
        self.model_size = model_size

    def _load_whisper(self):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"[AudioProcessor] Initializing faster-whisper model ({self.model_size}, INT8 CPU)")
            return WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as e:
            logger.error(f"[AudioProcessor] Failed to load faster-whisper: {e}")
            return None

    def extract_audio(self, video_path: str, output_audio_path: str) -> bool:
        """Extract 16kHz mono WAV audio track using ffmpeg."""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                output_audio_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return result.returncode == 0 and os.path.exists(output_audio_path)
        except Exception as e:
            logger.warning(f"[AudioProcessor] ffmpeg audio extraction failed: {e}")
            return False

    def transcribe_video(self, video_path: str) -> Dict[str, Any]:
        """
        Extract audio track and transcribe speech to text.
        Returns dict with 'text', 'segments', and 'language'.
        """
        if not os.path.exists(video_path):
            return {"text": "", "error": "Video file not found"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_path = tmp.name

        try:
            extracted = self.extract_audio(video_path, audio_path)
            if not extracted:
                return {"text": "", "error": "Could not extract audio track"}

            model = model_manager.get_model("whisper", self._load_whisper)
            if not model:
                return {"text": "", "error": "Whisper ASR model unavailable"}

            segments, info = model.transcribe(audio_path, beam_size=1, word_timestamps=False)
            
            transcribed_parts = []
            segment_data = []
            for segment in segments:
                transcribed_parts.append(segment.text.strip())
                segment_data.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip()
                })

            full_text = " ".join(transcribed_parts).strip()
            logger.info(f"[AudioProcessor] Transcribed {len(full_text)} chars from audio (Lang: {info.language})")

            return {
                "text": full_text,
                "language": info.language,
                "segments": segment_data,
                "error": None
            }
        except Exception as e:
            logger.error(f"[AudioProcessor] Transcription failed: {e}")
            return {"text": "", "error": str(e)}
        finally:
            if os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

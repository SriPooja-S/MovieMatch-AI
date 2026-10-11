import os
import re
from pathlib import Path
from typing import Optional, Dict
from app.utils.logger import logger


class URLProcessor:
    """
    Process video URLs from YouTube, Instagram, TikTok, Vimeo etc.
    Robust fallback: if download/metadata fails, extract info from URL title
    and let the user's query drive search instead.
    """

    PATTERNS = {
        "youtube": r"(youtube\.com|youtu\.be)",
        "instagram": r"instagram\.com",
        "twitter": r"(twitter\.com|x\.com)",
        "tiktok": r"tiktok\.com",
        "vimeo": r"vimeo\.com",
    }

    def __init__(self, download_dir: str = "temp_downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def validate_url(self, url: str) -> bool:
        url = url.strip()
        return url.startswith("http://") or url.startswith("https://")

    def detect_platform(self, url: str) -> Optional[str]:
        for platform, pattern in self.PATTERNS.items():
            if re.search(pattern, url, re.IGNORECASE):
                return platform
        return None

    def get_video_metadata(self, url: str) -> Optional[Dict]:
        """
        Try yt-dlp metadata extraction. Works for public videos.
        YouTube age-restricted / sign-in required videos will fail gracefully.
        """
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "skip_download": True,
                # Don't fail on geo-restrictions
                "ignoreerrors": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return {
                        "title": info.get("title", ""),
                        "description": info.get("description", ""),
                        "duration": info.get("duration", 0),
                        "uploader": info.get("uploader", ""),
                        "upload_date": info.get("upload_date", ""),
                        "view_count": info.get("view_count", 0),
                        "thumbnail": info.get("thumbnail", ""),
                        "platform": self.detect_platform(url),
                    }
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
        return None

    def download_video(self, url: str, quality: str = "worst") -> Optional[str]:
        """
        Download video for frame analysis.
        Returns path on success, None on failure (caller should fallback gracefully).
        """
        try:
            import yt_dlp
            platform = self.detect_platform(url)
            vid_id = self._extract_video_id(url, platform) or "video"
            output_template = str(self.download_dir / f"{vid_id}.%(ext)s")

            ydl_opts = {
                "format": "worst[ext=mp4]/worst",
                "outtmpl": output_template,
                "quiet": True,
                "no_warnings": True,
                "max_filesize": 50 * 1024 * 1024,  # 50MB
            }

            logger.info(f"Downloading video from {platform}: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info:
                    filename = ydl.prepare_filename(info)
                    if os.path.exists(filename):
                        logger.info(f"Downloaded: {filename}")
                        return filename
        except Exception as e:
            logger.error(f"Failed to download video: {e}")
        return None

    def extract_query_from_url(self, url: str) -> str:
        """
        Last-resort: extract a usable search query directly from the URL string.
        e.g. https://youtube.com/watch?v=XXX → try to get the title from URL params
        or just use the path segments.
        """
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(url)
            # YouTube search query param
            qs = parse_qs(parsed.query)
            if "search_query" in qs:
                return qs["search_query"][0]
            # Path-based (e.g. TikTok video titles in path)
            path = parsed.path.replace("/", " ").replace("-", " ").strip()
            return path if len(path) > 3 else ""
        except Exception:
            return ""

    def process_url(self, url: str) -> Dict:
        """
        Full processing pipeline with graceful degradation.
        Returns a dict with: platform, metadata, search_query, error (if any).
        """
        url = url.strip()
        if not self.validate_url(url):
            return {"error": "Invalid URL — must start with http:// or https://"}

        platform = self.detect_platform(url)
        result = {"platform": platform, "metadata": None, "search_query": "", "error": None}

        # Try metadata first
        metadata = self.get_video_metadata(url)
        if metadata:
            result["metadata"] = metadata
            # Build query from title + first 300 chars of description
            parts = []
            if metadata.get("title"):
                parts.append(metadata["title"])
            if metadata.get("description"):
                parts.append(metadata["description"][:300])
            result["search_query"] = " ".join(parts)
        else:
            # Fallback: parse query from URL itself
            fallback_query = self.extract_query_from_url(url)
            result["search_query"] = fallback_query
            if not fallback_query:
                result["error"] = (
                    "Could not fetch video metadata. "
                    "The video may require sign-in or be geo-restricted. "
                    "Please describe the video content manually in the text field below."
                )

        return result

    def cleanup(self):
        try:
            for f in self.download_dir.glob("*"):
                if f.is_file():
                    f.unlink()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _extract_video_id(self, url: str, platform: Optional[str]) -> Optional[str]:
        try:
            if platform == "youtube":
                for pattern in [r"v=([a-zA-Z0-9_-]{11})", r"youtu\.be/([a-zA-Z0-9_-]{11})", r"shorts/([a-zA-Z0-9_-]{11})"]:
                    m = re.search(pattern, url)
                    if m:
                        return m.group(1)
            elif platform == "instagram":
                m = re.search(r"instagram\.com/(?:p|reel)/([^/?]+)", url)
                if m:
                    return m.group(1)
        except Exception:
            pass
        return None

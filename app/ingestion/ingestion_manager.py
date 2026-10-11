import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from app.ingestion.tmdb_client import TMDBClient, GENRE_MAP
from app.ingestion.data_processor import DataProcessor
from app.core.embedder import EmbeddingGenerator
from app.utils.config import settings
from app.utils.logger import logger


class IngestionManager:
    """
    Manages the complete data ingestion pipeline.
    Supports full ingestion, targeted updates (by language/genre/year),
    and incremental updates.
    """

    def __init__(self):
        self.tmdb_client = TMDBClient()
        self.processor = DataProcessor()
        self.embedder = EmbeddingGenerator()
        self.movies_file = settings.MOVIES_FILE
        self.embeddings_file = settings.EMBEDDINGS_FILE
        self.metadata_file = settings.METADATA_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_ingest(
        self,
        fetch_details: bool = False,
        include_languages: List[str] = None,
        include_genres: List[int] = None,
        year_range: tuple = None
    ) -> Dict:
        """Complete re-ingestion with optional targeting."""
        logger.info("Starting full ingestion")
        start = datetime.now()

        raw = self.tmdb_client.fetch_comprehensive_dataset(
            include_languages=include_languages,
            include_genres=include_genres,
            year_range=year_range
        )

        self.processor.reset()
        processed = self.processor.process_movies(
            raw,
            fetch_details=fetch_details,
            tmdb_client=self.tmdb_client if fetch_details else None
        )

        texts = [m["text"] for m in processed]
        embeddings = self.embedder.generate_embeddings(texts)

        self._save_movies(processed)
        self._save_embeddings(embeddings)
        self._update_metadata(len(processed), "full_ingest")

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Full ingestion done: {len(processed)} movies in {duration:.1f}s")
        return {"total_movies": len(processed), "duration_seconds": duration}

    def incremental_update(self, days: int = 7) -> Dict:
        """Add recently released movies to the database."""
        logger.info(f"Starting incremental update (last {days} days)")
        start = datetime.now()

        existing = self._load_movies()

        # Fetch recent
        recent_raw = self.tmdb_client.fetch_recent_updates(days=days)
        recent_raw += self.tmdb_client.fetch_now_playing(pages=3)
        recent_raw += self.tmdb_client.fetch_upcoming(pages=3)

        self.processor.reset()
        for m in existing:
            self.processor.seen_ids.add(m["id"])
            self.processor.seen_titles.add(m.get("title", "").lower().strip())

        new_movies = self.processor.process_movies(recent_raw)
        new_count = len(new_movies)

        if new_count == 0:
            logger.info("No new movies to add")
            return {"new_movies": 0, "total_movies": len(existing)}

        all_movies = existing + new_movies
        texts = [m["text"] for m in all_movies]
        embeddings = self.embedder.generate_embeddings(texts)

        self._save_movies(all_movies)
        self._save_embeddings(embeddings)
        self._update_metadata(len(all_movies), "incremental_update",
                              {"new_movies": new_count, "days": days})

        duration = (datetime.now() - start).total_seconds()
        logger.info(f"Incremental update done: {new_count} new, {len(all_movies)} total in {duration:.1f}s")
        return {"new_movies": new_count, "total_movies": len(all_movies), "duration_seconds": duration}

    def targeted_update(
        self,
        languages: List[str] = None,
        genres: List[str] = None,
        year_min: int = None,
        year_max: int = None,
        pages: int = 5
    ) -> Dict:
        """
        Fetch and add movies matching specific criteria.
        Users can target specific languages, genres, or year ranges.
        """
        logger.info(f"Starting targeted update: lang={languages}, genre={genres}, years={year_min}-{year_max}")
        start = datetime.now()

        existing = self._load_movies()
        self.processor.reset()
        for m in existing:
            self.processor.seen_ids.add(m["id"])
            self.processor.seen_titles.add(m.get("title", "").lower().strip())

        raw = []

        # Fetch by language
        if languages:
            for lang in languages:
                raw += self.tmdb_client.fetch_movies_by_language(lang, pages=pages)

        # Fetch by genre
        if genres:
            # Build reverse genre name → id map
            name_to_id = {v.lower(): k for k, v in GENRE_MAP.items()}
            for genre_name in genres:
                gid = name_to_id.get(genre_name.lower())
                if gid:
                    raw += self.tmdb_client.fetch_movies_by_genre(gid, pages=pages)

        # Fetch by year range
        if year_min and year_max:
            for year in range(year_min, year_max + 1):
                raw += self.tmdb_client.fetch_movies_by_year(year, pages=2)
        elif year_min:
            from datetime import datetime as dt
            for year in range(year_min, dt.now().year + 1):
                raw += self.tmdb_client.fetch_movies_by_year(year, pages=2)

        # If nothing specified, fallback to latest releases
        if not raw:
            raw = self.tmdb_client.fetch_now_playing(pages=pages)
            raw += self.tmdb_client.fetch_upcoming(pages=pages)

        new_movies = self.processor.process_movies(raw)
        new_count = len(new_movies)

        if new_count == 0:
            return {"new_movies": 0, "total_movies": len(existing)}

        all_movies = existing + new_movies
        texts = [m["text"] for m in all_movies]
        embeddings = self.embedder.generate_embeddings(texts)

        self._save_movies(all_movies)
        self._save_embeddings(embeddings)
        self._update_metadata(len(all_movies), "targeted_update",
                              {"new_movies": new_count})

        duration = (datetime.now() - start).total_seconds()
        return {"new_movies": new_count, "total_movies": len(all_movies), "duration_seconds": duration}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_movies(self) -> List[Dict]:
        if not self.movies_file.exists():
            return []
        try:
            with open(self.movies_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load movies: {e}")
            return []

    def _save_movies(self, movies: List[Dict]):
        import numpy as np
        try:
            with open(self.movies_file, "w", encoding="utf-8") as f:
                json.dump(movies, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(movies)} movies to {self.movies_file}")
        except Exception as e:
            logger.error(f"Failed to save movies: {e}")
            raise

    def _save_embeddings(self, embeddings):
        import numpy as np
        try:
            np.save(self.embeddings_file, embeddings)
            logger.info(f"Saved embeddings to {self.embeddings_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise

    def _update_metadata(self, total: int, operation: str, details: Dict = None):
        metadata = {
            "last_update": datetime.now().isoformat(),
            "total_movies": total,
            "last_operation": operation,
            "details": details or {}
        }
        existing_meta = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    existing_meta = json.load(f)
            except Exception:
                pass

        metadata["version"] = existing_meta.get("version", 0) + 1
        history = existing_meta.get("history", [])
        history.append({
            "timestamp": metadata["last_update"],
            "operation": operation,
            "total_movies": total
        })
        metadata["history"] = history[-50:]

        try:
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    def get_metadata(self) -> Optional[Dict]:
        if not self.metadata_file.exists():
            return None
        try:
            with open(self.metadata_file) as f:
                return json.load(f)
        except Exception:
            return None

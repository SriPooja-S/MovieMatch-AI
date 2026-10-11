import json
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from app.core.embedder import EmbeddingGenerator
from app.utils.config import settings
from app.utils.logger import logger


from app.core.vector_db import VectorStore

class MovieSearchEngine:
    """
    Semantic search engine for movies using vector embeddings & FAISS-CPU.
    Supports text, image-derived, audio-transcribed, and multimodal queries.
    All embeddings are L2-normalized so dot product == cosine similarity.
    """

    def __init__(self):
        self.movies: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self._load_data()

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def _load_data(self):
        try:
            if settings.MOVIES_FILE.exists():
                with open(settings.MOVIES_FILE, "r", encoding="utf-8") as f:
                    self.movies = json.load(f)
                logger.info(f"Loaded {len(self.movies)} movies")
            else:
                logger.warning("Movies file not found")
                self.movies = []

            if settings.EMBEDDINGS_FILE.exists():
                self.embeddings = np.load(settings.EMBEDDINGS_FILE)
                logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")
            else:
                logger.warning("Embeddings file not found")
                self.embeddings = None

            if self.movies and self.embeddings is not None:
                if len(self.movies) != len(self.embeddings):
                    logger.error(
                        f"Data Desync Error: {len(self.movies)} movies vs "
                        f"{len(self.embeddings)} embeddings — enforcing strict alignment"
                    )
                    min_len = min(len(self.movies), len(self.embeddings))
                    self.movies = self.movies[:min_len]
                    self.embeddings = self.embeddings[:min_len]

                # Build FAISS / Vector Index
                self.vector_store.build_index(self.embeddings)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.movies = []
            self.embeddings = None

    def reload(self):
        logger.info("Reloading search engine data")
        self._load_data()

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = None,
        min_score: float = 0.0,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Semantic search using cosine similarity against pre-computed embeddings.
        Returns results ranked by match score (0-100).
        """
        if not self.movies or self.embeddings is None:
            logger.warning("No data loaded for search")
            return []

        top_k = top_k or settings.TOP_K_RESULTS

        # Encode query
        query_emb = self.embedder.generate_single_embedding(query)

        # Fast vector search via FAISS / VectorStore
        scores, top_indices = self.vector_store.search(query_emb, top_k=top_k * 5 if filters else top_k)

        results = []
        for score, idx in zip(scores, top_indices):
            if idx < 0 or idx >= len(self.movies):
                continue
            if score < min_score:
                continue

            movie = self.movies[idx]

            if filters and not self._matches_filters(movie, filters):
                continue

            results.append(self._format_result(movie, float(score)))

            if len(results) >= top_k:
                break

        logger.info(f"Search '{query[:60]}' → {len(results)} results")
        return results

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        min_score: float = 0.0,
        filters: Dict = None
    ) -> List[Dict]:
        """
        Search directly using a pre-computed embedding (for image/video search).
        """
        if not self.movies or self.embeddings is None:
            return []

        top_k = top_k or settings.TOP_K_RESULTS

        # Normalize input embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query_embedding = query_embedding.reshape(1, -1)

        scores = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx]) * 100
            if score < min_score:
                continue
            movie = self.movies[idx]
            if filters and not self._matches_filters(movie, filters):
                continue
            results.append(self._format_result(movie, score))
            if len(results) >= top_k:
                break

        return results

    def get_by_id(self, movie_id: int) -> Optional[Dict]:
        for movie in self.movies:
            if movie.get("id") == movie_id:
                return movie
        return None

    def get_similar(self, movie_id: int, top_k: int = 5) -> List[Dict]:
        """Find movies similar to a given movie ID."""
        # Find the index of this movie
        for idx, movie in enumerate(self.movies):
            if movie.get("id") == movie_id:
                query_emb = self.embeddings[idx]
                scores = np.dot(self.embeddings, query_emb.reshape(-1, 1)).flatten()
                top_indices = np.argsort(scores)[::-1][1:top_k + 1]  # skip self
                return [
                    self._format_result(self.movies[i], float(scores[i]) * 100)
                    for i in top_indices
                ]
        return []

    def browse(
        self,
        page: int = 1,
        page_size: int = 20,
        filters: Dict = None,
        sort_by: str = "rating"
    ) -> Dict:
        """Browse all movies with filtering and pagination."""
        filtered = []
        for movie in self.movies:
            if filters and not self._matches_filters(movie, filters):
                continue
            filtered.append(movie)

        # Sort
        if sort_by == "rating":
            filtered.sort(key=lambda m: m.get("vote_average", 0), reverse=True)
        elif sort_by == "year":
            filtered.sort(
                key=lambda m: m.get("release_date", "0000"),
                reverse=True
            )
        elif sort_by == "title":
            filtered.sort(key=lambda m: m.get("title", ""))

        total = len(filtered)
        start = (page - 1) * page_size
        end = start + page_size
        page_movies = [self._format_result(m, None) for m in filtered[start:end]]

        return {
            "movies": page_movies,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size)
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_lang(self, movie: Dict) -> str:
        """Get language code, supporting both field name variants."""
        return (movie.get("original_language") or movie.get("language") or "en").lower()

    def _format_result(self, movie: Dict, score) -> Dict:
        return {
            "id": movie.get("id"),
            "title": movie.get("title", "Unknown"),
            "original_title": movie.get("original_title", ""),
            "overview": movie.get("overview", ""),
            "score": round(score, 2) if score is not None else None,
            "genres": movie.get("genres", []),
            "release_date": movie.get("release_date", ""),
            "rating": movie.get("vote_average", 0) or 0,
            "vote_count": movie.get("vote_count", 0) or 0,
            "poster_path": movie.get("poster_path"),
            "backdrop_path": movie.get("backdrop_path"),
            "cast": movie.get("cast", []),
            "director": movie.get("director"),
            "original_language": self._get_lang(movie),
            "popularity": movie.get("popularity", 0) or 0,
            "keywords": movie.get("keywords", []),
            "tagline": movie.get("tagline", ""),
        }

    def _matches_filters(self, movie: Dict, filters: Dict) -> bool:
        if "genre" in filters and filters["genre"]:
            movie_genres = [g.lower() for g in movie.get("genres", [])]
            if filters["genre"].lower() not in movie_genres:
                return False

        if "language" in filters and filters["language"]:
            if self._get_lang(movie) != filters["language"].lower():
                return False

        if "year_min" in filters and filters["year_min"]:
            rd = movie.get("release_date", "")
            try:
                year = int(rd[:4])
                if year < filters["year_min"]:
                    return False
            except Exception:
                return False

        if "year_max" in filters and filters["year_max"]:
            rd = movie.get("release_date", "")
            try:
                year = int(rd[:4])
                if year > filters["year_max"]:
                    return False
            except Exception:
                return False

        if "min_rating" in filters and filters["min_rating"]:
            if movie.get("vote_average", 0) < filters["min_rating"]:
                return False

        if "actor" in filters and filters["actor"]:
            target_actor = filters["actor"].lower()
            cast_list = [c.lower() for c in movie.get("cast", [])]
            if not any(target_actor in actor for actor in cast_list):
                return False

        return True

    def get_stats(self) -> Dict:
        if not self.movies:
            return {"total_movies": 0}

        genre_counts = {}
        lang_counts = {}
        decade_counts = {}

        for movie in self.movies:
            for g in movie.get("genres", []):
                genre_counts[g] = genre_counts.get(g, 0) + 1

            lang = self._get_lang(movie)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            rd = movie.get("release_date", "")
            if rd and len(rd) >= 4:
                try:
                    year = int(rd[:4])
                    decade = (year // 10) * 10
                    decade_counts[decade] = decade_counts.get(decade, 0) + 1
                except Exception:
                    pass

        return {
            "total_movies": len(self.movies),
            "genres": dict(sorted(genre_counts.items(), key=lambda x: -x[1])),
            "languages": dict(sorted(lang_counts.items(), key=lambda x: -x[1])[:20]),
            "decades": dict(sorted(decade_counts.items())),
            "embedding_dimension": (
                self.embeddings.shape[1] if self.embeddings is not None else 0
            ),
        }

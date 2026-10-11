import requests
import time
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from app.utils.config import settings
from app.utils.logger import logger

# TMDB genre ID map (saves API call for basic processing)
GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}

# Common cinema languages
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "ko": "Korean", "ja": "Japanese",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "zh": "Chinese", "pt": "Portuguese", "ru": "Russian", "ar": "Arabic",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "bn": "Bengali",
    "tr": "Turkish", "th": "Thai", "id": "Indonesian", "pl": "Polish",
}


class TMDBClient:
    """TMDB API client with comprehensive movie fetching and watch providers."""

    def __init__(self):
        self.api_key = settings.TMDB_API_KEY
        self.base_url = settings.TMDB_BASE_URL
        self.session = self._create_session()
        if not self.api_key:
            raise ValueError("TMDB_API_KEY not configured")

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=settings.MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _make_request(self, endpoint: str, params: Dict = None, lang: str = "en-US") -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint}"
        p = {"api_key": self.api_key, "language": lang}
        if params:
            p.update(params)
        try:
            resp = self.session.get(url, params=p, timeout=settings.FETCH_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            return None

    def _fetch_paginated(self, endpoint: str, pages: int, extra_params: Dict = None) -> List[Dict]:
        movies = []
        for page in range(1, pages + 1):
            params = {"page": page}
            if extra_params:
                params.update(extra_params)
            data = self._make_request(endpoint, params)
            if data and "results" in data:
                movies.extend(data["results"])
            else:
                logger.warning(f"No results from {endpoint} page {page}")
                break
            time.sleep(0.1)  # Rate limiting
        return movies

    def fetch_popular_movies(self, pages: int = 10) -> List[Dict]:
        logger.info(f"Fetching {pages} pages of popular movies")
        return self._fetch_paginated("movie/popular", pages)

    def fetch_top_rated_movies(self, pages: int = 10) -> List[Dict]:
        logger.info(f"Fetching {pages} pages of top-rated movies")
        return self._fetch_paginated("movie/top_rated", pages)

    def fetch_now_playing(self, pages: int = 5) -> List[Dict]:
        logger.info(f"Fetching {pages} pages of now playing movies")
        return self._fetch_paginated("movie/now_playing", pages)

    def fetch_upcoming(self, pages: int = 5) -> List[Dict]:
        logger.info(f"Fetching {pages} pages of upcoming movies")
        return self._fetch_paginated("movie/upcoming", pages)

    def fetch_movies_by_genre(self, genre_id: int, pages: int = 5, language: str = None) -> List[Dict]:
        params = {"with_genres": genre_id, "sort_by": "popularity.desc"}
        if language:
            params["with_original_language"] = language
        return self._fetch_paginated("discover/movie", pages, params)

    def fetch_movies_by_language(self, language_code: str, pages: int = 5) -> List[Dict]:
        """Fetch movies in a specific original language."""
        params = {
            "with_original_language": language_code,
            "sort_by": "popularity.desc"
        }
        return self._fetch_paginated("discover/movie", pages, params)

    def fetch_movies_by_year(self, year: int, pages: int = 3) -> List[Dict]:
        params = {"primary_release_year": year, "sort_by": "popularity.desc"}
        return self._fetch_paginated("discover/movie", pages, params)

    def fetch_movies_by_actor(self, actor_id: int, pages: int = 3) -> List[Dict]:
        params = {"with_cast": actor_id, "sort_by": "popularity.desc"}
        return self._fetch_paginated("discover/movie", pages, params)

    def fetch_movie_details(self, movie_id: int) -> Optional[Dict]:
        """Full movie details including credits, keywords, watch providers."""
        params = {"append_to_response": "credits,keywords,watch/providers"}
        return self._make_request(f"movie/{movie_id}", params)

    def fetch_watch_providers(self, movie_id: int, region: str = "US") -> Dict:
        """Get streaming availability for a movie."""
        data = self._make_request(f"movie/{movie_id}/watch/providers")
        if not data:
            return {}
        results = data.get("results", {})
        region_data = results.get(region, {})
        providers = {}
        if "flatrate" in region_data:
            providers["streaming"] = [
                {"name": p["provider_name"], "logo": p.get("logo_path")}
                for p in region_data["flatrate"]
            ]
        if "rent" in region_data:
            providers["rent"] = [
                {"name": p["provider_name"], "logo": p.get("logo_path")}
                for p in region_data["rent"]
            ]
        if "buy" in region_data:
            providers["buy"] = [
                {"name": p["provider_name"], "logo": p.get("logo_path")}
                for p in region_data["buy"]
            ]
        providers["link"] = region_data.get("link", "")
        return providers

    def search_movies(self, query: str, pages: int = 1) -> List[Dict]:
        """Direct title-based search on TMDB."""
        return self._fetch_paginated("search/movie", pages, {"query": query})

    def fetch_all_genres(self) -> List[Dict]:
        data = self._make_request("genre/movie/list")
        return data.get("genres", []) if data else []

    def fetch_recent_updates(self, days: int = 7) -> List[Dict]:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        params = {
            "primary_release_date.gte": start_date.strftime("%Y-%m-%d"),
            "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
            "sort_by": "primary_release_date.desc"
        }
        logger.info(f"Fetching movies from last {days} days")
        return self._fetch_paginated("discover/movie", pages=5, extra_params=params)

    def fetch_comprehensive_dataset(
        self,
        include_languages: List[str] = None,
        include_genres: List[int] = None,
        year_range: tuple = None
    ) -> List[Dict]:
        """Fetch a comprehensive, configurable dataset."""
        all_movies = []

        all_movies.extend(self.fetch_popular_movies(pages=10))
        all_movies.extend(self.fetch_top_rated_movies(pages=10))
        all_movies.extend(self.fetch_now_playing(pages=5))
        all_movies.extend(self.fetch_upcoming(pages=5))

        # Decade coverage
        current_year = datetime.now().year
        start_year = year_range[0] if year_range else 1980
        end_year = year_range[1] if year_range else current_year
        for year in range(start_year, end_year + 1, 5):
            all_movies.extend(self.fetch_movies_by_year(year, pages=3))

        # Classic cinema
        for year in [1970, 1960, 1950, 1940]:
            if year >= start_year:
                all_movies.extend(self.fetch_movies_by_year(year, pages=2))

        # Genre coverage
        genres = include_genres or [g for g in GENRE_MAP.keys()]
        for genre_id in genres[:15]:
            all_movies.extend(self.fetch_movies_by_genre(genre_id, pages=3))

        # Language coverage
        languages = include_languages or ["hi", "ko", "ja", "fr", "es", "zh", "ta", "te"]
        for lang in languages:
            all_movies.extend(self.fetch_movies_by_language(lang, pages=5))

        logger.info(f"Fetched {len(all_movies)} total movies (before dedup)")
        return all_movies

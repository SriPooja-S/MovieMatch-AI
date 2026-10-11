from typing import List, Dict, Set, Optional
from app.ingestion.tmdb_client import GENRE_MAP
from app.utils.logger import logger


class DataProcessor:
    """
    Process and enrich movie data from TMDB.
    Creates rich combined text fields for high-quality semantic embeddings.
    """

    def __init__(self):
        self.seen_ids: Set[int] = set()
        self.seen_titles: Set[str] = set()

    def reset(self):
        self.seen_ids.clear()
        self.seen_titles.clear()

    def process_movies(
        self,
        raw_movies: List[Dict],
        fetch_details: bool = False,
        tmdb_client=None
    ) -> List[Dict]:
        processed = []
        for movie in raw_movies:
            try:
                if not self._validate(movie):
                    continue

                movie_id = movie["id"]
                title = movie.get("title", "").strip()
                norm_title = title.lower().strip()

                if movie_id in self.seen_ids or norm_title in self.seen_titles:
                    continue

                self.seen_ids.add(movie_id)
                self.seen_titles.add(norm_title)

                if fetch_details and tmdb_client:
                    detailed = tmdb_client.fetch_movie_details(movie_id)
                    if detailed:
                        movie = detailed

                enriched = self._enrich(movie)
                processed.append(enriched)

            except Exception as e:
                logger.warning(f"Failed to process movie {movie.get('id')}: {e}")

        logger.info(f"Processed {len(processed)} unique movies")
        return processed

    def _validate(self, movie: Dict) -> bool:
        return bool(movie.get("id") and movie.get("title"))

    def _enrich(self, movie: Dict) -> Dict:
        # Genres
        genres = []
        if "genres" in movie:
            genres = [g["name"] for g in movie.get("genres", []) if isinstance(g, dict)]
        elif "genre_ids" in movie:
            genres = [GENRE_MAP.get(gid, "") for gid in movie.get("genre_ids", [])]
            genres = [g for g in genres if g]

        # Cast (top 5)
        cast = []
        if "credits" in movie and "cast" in movie["credits"]:
            cast = [a["name"] for a in movie["credits"]["cast"][:5]]

        # Director
        director = None
        if "credits" in movie and "crew" in movie["credits"]:
            for person in movie["credits"]["crew"]:
                if person.get("job") == "Director":
                    director = person["name"]
                    break

        # Keywords
        keywords = []
        if "keywords" in movie:
            kw_data = movie["keywords"]
            if isinstance(kw_data, dict):
                keywords = [k["name"] for k in kw_data.get("keywords", [])]
            elif isinstance(kw_data, list):
                keywords = kw_data

        # Tagline
        tagline = movie.get("tagline", "")

        # Build rich text for embedding
        # Order matters: title → tagline → overview → genres → cast → keywords
        text_parts = []

        title = movie.get("title", "")
        original_title = movie.get("original_title", "")
        overview = movie.get("overview", "")

        text_parts.append(f"Title: {title}")
        if original_title and original_title != title:
            text_parts.append(f"Original title: {original_title}")
        if tagline:
            text_parts.append(f"Tagline: {tagline}")
        if overview:
            text_parts.append(f"Plot: {overview}")
        if genres:
            text_parts.append(f"Genres: {', '.join(genres)}")
        if cast:
            text_parts.append(f"Cast: {', '.join(cast)}")
        if director:
            text_parts.append(f"Director: {director}")
        if keywords:
            text_parts.append(f"Themes: {', '.join(keywords[:15])}")

        release_date = movie.get("release_date", "")
        lang = movie.get("original_language", "en")
        if release_date:
            text_parts.append(f"Year: {release_date[:4]}")
        text_parts.append(f"Language: {lang}")

        text = ". ".join(text_parts)

        return {
            "id": movie["id"],
            "title": title,
            "original_title": original_title,
            "overview": overview,
            "tagline": tagline,
            "genres": genres,
            "cast": cast,
            "director": director,
            "keywords": keywords[:20],
            "release_date": release_date,
            "vote_average": round(movie.get("vote_average", 0), 1),
            "vote_count": movie.get("vote_count", 0),
            "popularity": movie.get("popularity", 0),
            "poster_path": movie.get("poster_path"),
            "backdrop_path": movie.get("backdrop_path"),
            "original_language": lang,
            "text": text,  # This is what gets embedded
        }

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # TMDB API
    TMDB_API_KEY: str = "8e82500b53ef7f9327135067fb6f789f"
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE: str = "https://image.tmdb.org/t/p"

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "app" / "data"
    MOVIES_FILE: Path = DATA_DIR / "movies.json"
    EMBEDDINGS_FILE: Path = DATA_DIR / "embeddings.npy"
    METADATA_FILE: Path = DATA_DIR / "metadata.json"
    VIDEO_DOWNLOAD_DIR: str = "temp_downloads"

    # Search
    MIN_SIMILARITY_SCORE: float = 0.0
    TOP_K_RESULTS: int = 10

    # Models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL: str = "ViT-B/32"

    # Ingestion
    TMDB_PAGES_PER_ENDPOINT: int = 10
    FETCH_TIMEOUT: int = 30
    MAX_RETRIES: int = 5
    BATCH_SIZE: int = 100

    # Video / Image
    MAX_VIDEO_SIZE_MB: int = 50
    DEFAULT_FRAME_EXTRACTION: int = 10
    MAX_IMAGE_SIZE_MB: int = 10

    # OCR
    ENABLE_OCR: bool = True
    OCR_LANGUAGE: str = "eng"

    # Hardware
    USE_GPU: str = "auto"

    # Cache
    CACHE_ENABLED: bool = True

    # Scheduling
    UPDATE_SCHEDULE: str = "0 2 * * *"
    FULL_REFRESH_SCHEDULE: str = "0 3 * * 0"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "app.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    def model_post_init(self, __context):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)


settings = Settings()

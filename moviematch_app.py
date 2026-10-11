"""
MovieMatch AI — Multimodal Movie Discovery Engine
Single-file Streamlit app: all tabs unified, no broken page routing.
"""

import sys
import io
import time
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.core.engine import MovieSearchEngine
from app.ingestion.ingestion_manager import IngestionManager
from app.ingestion.tmdb_client import GENRE_MAP, LANGUAGE_NAMES
from app.processors.image_processor import ImageProcessor
from app.processors.url_processor import URLProcessor
from app.processors.video_processor import VideoProcessor
from app.utils.config import settings
from app.utils.logger import logger

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieMatch AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  [data-testid="stAppViewContainer"] { background: #0d0d14; }
  [data-testid="stSidebar"] { background: #13131f; border-right: 1px solid #2a2a3d; }
  h1, h2, h3, h4 { color: #f0f0ff !important; }
  p, li, label, .stMarkdown { color: #c0c0d8 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #1a1a2e; border-radius: 10px; padding: 8px 12px;
    border: 1px solid #2a2a4a;
  }

  /* Primary button */
  div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e50914 0%, #b0060f 100%);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; font-size: 15px; padding: 10px 20px;
    transition: all 0.2s;
  }
  div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px); box-shadow: 0 4px 20px #e5091460;
  }

  /* Secondary button */
  div.stButton > button {
    background: #1e1e30; color: #d0d0f0; border: 1px solid #3a3a5a;
    border-radius: 8px;
  }

  /* Input fields */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {
    background: #1a1a2e !important; color: #e0e0ff !important;
    border: 1px solid #3a3a5a !important; border-radius: 8px !important;
  }

  /* Movie card */
  .movie-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a4a; border-radius: 14px;
    padding: 16px; margin-bottom: 16px;
    transition: border-color 0.2s;
  }
  .movie-card:hover { border-color: #e50914; }
  .movie-title { font-size: 18px; font-weight: 700; color: #ffffff !important; }
  .movie-meta { color: #8888aa !important; font-size: 13px; }
  .score-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-weight: 700; font-size: 13px;
  }
  .score-high { background: #0d3d1a; color: #4ade80; border: 1px solid #4ade80; }
  .score-mid  { background: #3d2e0d; color: #fbbf24; border: 1px solid #fbbf24; }
  .score-low  { background: #3d0d0d; color: #f87171; border: 1px solid #f87171; }

  /* OTT provider pill */
  .provider-pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
    background: #252540; color: #aaaacc; border: 1px solid #3a3a5a;
    margin: 2px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #13131f; border-radius: 10px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #8888aa; border-radius: 8px; padding: 8px 20px; }
  .stTabs [aria-selected="true"] {
    background: #e50914 !important; color: white !important; font-weight: 700;
  }

  /* Slider */
  .stSlider [data-baseweb="slider"] { color: #e50914; }

  /* Expander */
  .streamlit-expanderHeader { background: #1a1a2e !important; color: #c0c0e0 !important; }
  .streamlit-expanderContent { background: #13131f !important; }

  /* Section divider */
  .section-divider { border: none; border-top: 1px solid #2a2a4a; margin: 16px 0; }

  /* Tag */
  .genre-tag {
    display: inline-block; background: #1e1e38; color: #9999cc;
    border: 1px solid #3a3a58; border-radius: 6px;
    padding: 2px 8px; font-size: 12px; margin: 2px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_search_engine():
    return MovieSearchEngine()

@st.cache_resource(show_spinner=False)
def get_image_processor():
    return ImageProcessor()

@st.cache_resource(show_spinner=False)
def get_url_processor():
    return URLProcessor()

@st.cache_resource(show_spinner=False)
def get_video_processor():
    return VideoProcessor()

@st.cache_resource(show_spinner=False)
def get_multimodal_processor():
    try:
        from app.core.multimodal_processor import MultiModalProcessor
        return MultiModalProcessor()
    except Exception as e:
        logger.warning(f"CLIP not available: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_ingestion_manager():
    return IngestionManager()


# ── Helper functions ─────────────────────────────────────────────────────────
POSTER_BASE = "https://image.tmdb.org/t/p/w300"
POSTER_PLACEHOLDER = "https://placehold.co/300x450/1a1a2e/666699?text=No+Poster"

def poster_url(path):
    if path:
        return f"{POSTER_BASE}{path}"
    return POSTER_PLACEHOLDER

def score_badge(score):
    if score is None:
        return ""
    if score >= 70:
        cls = "score-high"
    elif score >= 45:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score:.1f}% match</span>'

def star_rating(rating):
    filled = int(round(rating / 2))
    return f"{rating:.1f}/10"

def language_name(code):
    return LANGUAGE_NAMES.get(code, code.upper() if code else "?")

def year_from_date(date_str):
    try:
        return date_str[:4] if date_str else "—"
    except Exception:
        return "—"

def render_movie_card(movie, idx=0, show_score=True):
    """Render a single movie card with poster, metadata, OTT providers."""
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col_img, col_info = st.columns([1, 4])

        with col_img:
            st.image(poster_url(movie.get("poster_path")), use_container_width=True)

        with col_info:
            # Title row
            title_html = f'<span class="movie-title">{movie["title"]}</span>'
            orig = movie.get("original_title", "")
            if orig and orig != movie["title"]:
                title_html += f' <span style="color:#666688;font-size:13px">({orig})</span>'
            if show_score and movie.get("score") is not None:
                title_html += " " + score_badge(movie["score"])
            st.markdown(title_html, unsafe_allow_html=True)

            # Meta row
            yr = year_from_date(movie.get("release_date", ""))
            lang = language_name(movie.get("original_language", ""))
            rating = movie.get("rating", 0) or 0
            votes = movie.get("vote_count", 0) or 0
            tagline = movie.get("tagline", "")

            meta = f'<span class="movie-meta">Year: {yr} &nbsp;|&nbsp; Lang: {lang} &nbsp;|&nbsp; Rating: {star_rating(rating)}'
            if votes:
                meta += f" ({votes:,} votes)"
            meta += "</span>"
            st.markdown(meta, unsafe_allow_html=True)

            if tagline:
                st.markdown(f'<i style="color:#8888aa;font-size:13px">"{tagline}"</i>', unsafe_allow_html=True)

            # Genres
            genres = movie.get("genres", [])
            if genres:
                genre_html = "".join(f'<span class="genre-tag">{g}</span>' for g in genres[:6])
                st.markdown(genre_html, unsafe_allow_html=True)

            # Director / Cast
            if movie.get("director"):
                st.markdown(f'<span class="movie-meta"><b>Director:</b> {movie["director"]}</span>', unsafe_allow_html=True)
            cast = movie.get("cast", [])
            if cast:
                st.markdown(f'<span class="movie-meta"><b>Cast:</b> {", ".join(cast[:4])}</span>', unsafe_allow_html=True)

            # Overview
            overview = movie.get("overview", "")
            if overview:
                short = overview[:220] + "…" if len(overview) > 220 else overview
                st.markdown(f'<p style="color:#aaaacc;font-size:13px;margin-top:6px">{short}</p>', unsafe_allow_html=True)

            # OTT providers (fetch live)
            _render_watch_providers(movie.get("id"), idx)

        st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_providers_cached(movie_id: int):
    try:
        from app.ingestion.tmdb_client import TMDBClient
        client = TMDBClient()
        return client.fetch_watch_providers(movie_id)
    except Exception:
        return {}


def _render_watch_providers(movie_id, idx):
    if not movie_id:
        return
    try:
        providers = _fetch_providers_cached(movie_id)
        streaming = providers.get("streaming", [])
        rent = providers.get("rent", [])
        link = providers.get("link", "")

        if streaming:
            names = " ".join(f'<span class="provider-pill">▶ {p["name"]}</span>' for p in streaming[:5])
            st.markdown(f'<div style="margin-top:6px"><b style="color:#aaaacc">Stream:</b> {names}</div>', unsafe_allow_html=True)
        elif rent:
            names = " ".join(f'<span class="provider-pill">{p["name"]}</span>' for p in rent[:3])
            st.markdown(f'<div style="margin-top:6px"><b style="color:#aaaacc">Rent:</b> {names}</div>', unsafe_allow_html=True)

        if link:
            st.markdown(f'<a href="{link}" target="_blank" style="font-size:12px;color:#6666cc">View all streaming options</a>', unsafe_allow_html=True)
    except Exception:
        pass


def render_results_grid(results, title="Results", show_score=True):
    if not results:
        st.warning("No movies found. Try a different query or relax your filters.")
        return
    st.markdown(f"### {title} — {len(results)} found")
    for idx, movie in enumerate(results):
        render_movie_card(movie, idx=idx, show_score=show_score)


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    engine = get_search_engine()
    stats = engine.get_stats()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:12px 0">
          <h2 style="color:#e50914!important;margin:0">MovieMatch AI</h2>
          <p style="color:#888899!important;font-size:12px">Multimodal Movie Discovery</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Stats
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Movies", f"{stats.get('total_movies', 0):,}")
        with c2:
            st.metric("Genres", len(stats.get("genres", {})))

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Quick actions
        st.markdown("**Quick Actions**")
        if st.button("Reload Database", use_container_width=True):
            get_search_engine.clear()
            st.cache_resource.clear()
            st.success("Database reloaded!")
            st.rerun()

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # Last update info
        try:
            mgr = get_ingestion_manager()
            meta = mgr.get_metadata()
            if meta:
                last_update = meta.get("last_update", "")[:10]
                st.markdown(f'<p style="color:#666688;font-size:12px">Last updated: {last_update}</p>', unsafe_allow_html=True)
        except Exception:
            pass

        # Top genres
        genres = stats.get("genres", {})
        if genres:
            top_genres = list(genres.items())[:8]
            st.markdown("**Top Genres**")
            for genre, count in top_genres:
                pct = count / stats["total_movies"] * 100 if stats["total_movies"] else 0
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#8888aa">'
                    f'<span>{genre}</span><span>{count}</span></div>',
                    unsafe_allow_html=True
                )
                st.progress(pct / 100)


# ── Tab 1: Text Search ───────────────────────────────────────────────────────
def tab_text_search():
    st.markdown("## Describe any movie in natural language")
    st.markdown(
        '<p style="color:#8888aa">Use plot descriptions, partial memories, themes, character traits, '
        'emotions — in any language.</p>',
        unsafe_allow_html=True
    )

    # Examples
    with st.expander("Example queries"):
        examples = [
            "A philosophical animated Pixar movie about the soul and purpose of life",
            "Hero experiences love in three stages of life, second love is heartbreaking even though they love each other, heroine forgets everything",
            "A girl travels through time using a diary to save someone she loves",
            "A robot who befriends a child and learns what it means to be human",
            "Korean thriller about class inequality where poor family infiltrates rich household",
            "Animated movie where a child enters a spirit world to save her parents",
            "Superhero who loses his powers and must discover true identity",
            "A man stuck in a time loop reliving the same day",
        ]
        for ex in examples:
            if st.button(ex[:80], key=f"ex_{ex[:20]}"):
                st.session_state["text_query_prefill"] = ex

    query = st.text_area(
        "Your query",
        value=st.session_state.get("text_query_prefill", ""),
        placeholder="e.g. A philosophical animated movie about music and the purpose of life…",
        height=100,
        key="text_query_input"
    )

    # Filters row
    with st.expander("Filters & Settings"):
        c1, c2, c3, c4 = st.columns(4)
        engine = get_search_engine()
        stats = engine.get_stats()

        with c1:
            num_results = st.slider("Results", 1, 20, 10)
            min_score = st.slider("Min match %", 0, 100, 0)
        with c2:
            genres_list = sorted(stats.get("genres", {}).keys())
            sel_genre = st.selectbox("Genre", ["Any"] + genres_list)
        with c3:
            lang_codes = sorted(stats.get("languages", {}).keys())
            lang_options = {language_name(c): c for c in lang_codes}
            sel_lang_name = st.selectbox("Language", ["Any"] + list(lang_options.keys()))
            sel_lang = lang_options.get(sel_lang_name)
        with c4:
            year_min = st.number_input("Year from", 1900, 2030, 1900)
            year_max = st.number_input("Year to", 1900, 2030, 2030)
            min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, 0.5)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a query.")
            return

        filters = {}
        if sel_genre and sel_genre != "Any":
            filters["genre"] = sel_genre
        if sel_lang_name and sel_lang_name != "Any" and sel_lang:
            filters["language"] = sel_lang
        if year_min > 1900:
            filters["year_min"] = year_min
        if year_max < 2030:
            filters["year_max"] = year_max
        if min_rating > 0:
            filters["min_rating"] = min_rating

        with st.spinner("Finding movies…"):
            engine = get_search_engine()
            results = engine.search(
                query=query.strip(),
                top_k=num_results,
                min_score=min_score,
                filters=filters or None,
            )

        render_results_grid(results, title="Search Results")


# ── Tab 2: Image Search ──────────────────────────────────────────────────────
def tab_image_search():
    st.markdown("## Search by uploading an image")
    st.markdown('<p style="color:#8888aa">Upload a movie screenshot, poster, or any scene — AI will analyse it and find matching movies.</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Movie poster, screenshot, or scene from any film"
    )

    if not uploaded:
        st.markdown("""
        <div style="background:#1a1a2e;border:2px dashed #3a3a5a;border-radius:14px;padding:40px;text-align:center;margin-top:20px">
          <p style="color:#8888aa;font-weight:600">Drag & drop or click to upload a movie image</p>
          <p style="color:#555577;font-size:13px">Works with: posters, screenshots, scene stills, TV captures</p>
        </div>
        """, unsafe_allow_html=True)
        return

    col_img, col_settings = st.columns([1, 2])
    image = Image.open(uploaded).convert("RGB")

    with col_img:
        st.markdown("**Uploaded Image**")
        st.image(image, use_container_width=True)
        st.markdown(f'<span class="movie-meta">Size: {image.width}×{image.height}px</span>', unsafe_allow_html=True)

    with col_settings:
        st.markdown("**Search Settings**")
        method = st.radio(
            "Method",
            ["CLIP Visual Analysis (Recommended)", "OCR Text Extraction", "Combined"],
            help="CLIP analyses visual content; OCR reads text in image; Combined uses both"
        )
        num_results = st.slider("Number of results", 1, 20, 10, key="img_n")
        min_score = st.slider("Min match %", 0, 100, 20, key="img_min")

        if st.button("Search for Similar Movies", type="primary", use_container_width=True):
            _run_image_search(image, method, num_results, min_score)


def _run_image_search(image, method, num_results, min_score):
    engine = get_search_engine()
    image_proc = get_image_processor()
    mm_proc = get_multimodal_processor()

    with st.spinner("Analysing image…"):
        features = image_proc.preprocess_for_search(image)
        ocr_text = features.get("text", "")

        query = ""
        used_clip = False

        if "CLIP" in method or "Combined" in method:
            if mm_proc and mm_proc.is_available:
                query = mm_proc.image_to_text_query(image, top_n=6)
                used_clip = True
            else:
                st.warning("CLIP model not available — falling back to OCR.")

        if ("OCR" in method or "Combined" in method or not query) and ocr_text:
            query = f"{query} {ocr_text}".strip() if query else ocr_text

        if not query:
            st.error("Could not extract features from this image. Try a different image or method.")
            return

    st.info(f"Generated query: *{query[:150]}*")

    with st.spinner("Searching database…"):
        results = engine.search(query=query, top_k=num_results, min_score=min_score)

    render_results_grid(results, title="Visual Match Results")


# ── Tab 3: Video / URL Search ────────────────────────────────────────────────
def tab_video_search():
    st.markdown("## Search using a video URL or upload")
    st.markdown('<p style="color:#8888aa">Paste a YouTube/Instagram/TikTok/Vimeo URL, or upload a video clip directly.</p>', unsafe_allow_html=True)

    input_method = st.radio("Input type", ["Video URL", "Upload Video File"], horizontal=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if input_method == "Video URL":
        _url_search_section()
    else:
        _upload_video_section()


def _url_search_section():
    url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=... or any trailer/clip URL",
    )

    c1, c2 = st.columns(2)
    with c1:
        num_results = st.slider("Results", 1, 20, 10, key="url_n")
        min_score = st.slider("Min match %", 0, 100, 20, key="url_min")
    with c2:
        use_frames = st.checkbox(
            "Download & analyse video frames",
            value=False,
            help="Slower but more accurate — extracts visual frames via CLIP"
        )

    if not st.button("Search", type="primary"):
        _show_url_examples()
        return

    if not url.strip():
        st.warning("Please enter a URL.")
        return

    url_proc = get_url_processor()
    engine = get_search_engine()
    video_proc = get_video_processor()
    mm_proc = get_multimodal_processor()

    with st.spinner("Fetching video information…"):
        data = url_proc.process_url(url.strip())

    platform = data.get("platform", "unknown")
    if platform:
        st.success(f"Platform detected: **{platform.upper()}**")

    metadata = data.get("metadata")
    if metadata:
        with st.expander("Video Information", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                st.write(f"**Uploader:** {metadata.get('uploader', 'N/A')}")
                st.write(f"**Duration:** {metadata.get('duration', 0)}s")
            with c2:
                st.write(f"**Views:** {metadata.get('view_count', 0):,}")
                st.write(f"**Upload date:** {metadata.get('upload_date', 'N/A')}")
            if metadata.get("thumbnail"):
                st.image(metadata["thumbnail"], width=300)

    search_query = data.get("search_query", "").strip()

    # If frames requested and CLIP available
    if use_frames and mm_proc and mm_proc.is_available:
        st.info("Downloading video frames (may take ~30–60s)…")
        video_path = url_proc.download_video(url.strip())
        if video_path:
            with st.spinner("Extracting frames…"):
                frames = video_proc.extract_frames(video_path, num_frames=8)
            if frames:
                with st.expander("Extracted Frames"):
                    cols = st.columns(min(len(frames), 4))
                    for i, fr in enumerate(frames[:4]):
                        with cols[i]:
                            st.image(fr, use_container_width=True, caption=f"Frame {i+1}")
                with st.spinner("Analysing visual content…"):
                    visual_query = mm_proc.images_to_text_query(frames, top_n=6)
                search_query = f"{search_query} {visual_query}".strip()
                st.success(f"Visual query: *{visual_query[:120]}*")
            url_proc.cleanup()
        else:
            st.warning("Could not download video. Using metadata for search.")

    # Error from URL processing
    if data.get("error") and not search_query:
        st.error(data["error"])
        st.markdown("**Alternatively, describe the video content below:**")
        manual = st.text_area("Manual description", height=80)
        if manual.strip() and st.button("Search with description", type="primary"):
            search_query = manual.strip()
        else:
            return

    if not search_query:
        st.error("No search query could be generated. Please describe the video content manually.")
        return

    st.info(f"Searching with: *{search_query[:150]}*")

    with st.spinner("Searching database…"):
        results = engine.search(query=search_query, top_k=num_results, min_score=min_score)

    render_results_grid(results, title="Video Match Results")


def _upload_video_section():
    uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm"])
    if not uploaded:
        return

    num_results = st.slider("Results", 1, 20, 10, key="vid_n")
    min_score = st.slider("Min match %", 0, 100, 20, key="vid_min")

    if not st.button("Analyse & Search", type="primary"):
        return

    import tempfile, os
    engine = get_search_engine()
    video_proc = get_video_processor()
    mm_proc = get_multimodal_processor()

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Extracting frames…"):
            frames = video_proc.extract_frames(tmp_path, num_frames=8)

        if not frames:
            st.error("Could not extract frames from this video.")
            return

        with st.expander("Extracted Frames"):
            cols = st.columns(min(len(frames), 4))
            for i, fr in enumerate(frames[:4]):
                with cols[i]:
                    st.image(fr, use_container_width=True, caption=f"Frame {i+1}")

        if mm_proc and mm_proc.is_available:
            with st.spinner("Analysing visual content…"):
                query = mm_proc.images_to_text_query(frames, top_n=6)
            st.info(f"Generated query: *{query[:150]}*")
        else:
            st.warning("CLIP not available — please describe the video manually.")
            query = st.text_input("Describe the video content:")
            if not query:
                return

        with st.spinner("Searching…"):
            results = engine.search(query=query, top_k=num_results, min_score=min_score)

        render_results_grid(results, title="Video Match Results")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _show_url_examples():
    st.markdown("""
    <div style="background:#1a1a2e;border-radius:12px;padding:20px;margin-top:16px">
      <h4 style="color:#9999cc">What works well</h4>
      <ul style="color:#8888aa;font-size:13px">
        <li>Movie trailers on YouTube</li>
        <li>Instagram reels with movie clips</li>
        <li>TikTok videos discussing movies</li>
        <li>Vimeo short films or clips</li>
      </ul>
      <h4 style="color:#9999cc">Known limitations</h4>
      <ul style="color:#8888aa;font-size:13px">
        <li>Age-restricted YouTube videos require sign-in (use metadata-only mode)</li>
        <li>Private or region-blocked videos may fail</li>
        <li>Very short Shorts/Reels may have minimal metadata</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 4: Explore Database ──────────────────────────────────────────────────
def tab_explore():
    st.markdown("## Explore Movie Database")
    st.markdown('<p style="color:#8888aa">Browse, filter, and search all movies in the local database.</p>', unsafe_allow_html=True)

    engine = get_search_engine()
    stats = engine.get_stats()

    # Filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            genres_list = sorted(stats.get("genres", {}).keys())
            sel_genre = st.selectbox("Genre", ["Any"] + genres_list, key="exp_genre")
        with c2:
            lang_codes = sorted(stats.get("languages", {}).keys())
            lang_options = {language_name(c): c for c in lang_codes}
            sel_lang_name = st.selectbox("Language", ["Any"] + list(lang_options.keys()), key="exp_lang")
            sel_lang = lang_options.get(sel_lang_name)
        with c3:
            year_min = st.number_input("Year from", 1900, 2030, 1900, key="exp_yr_min")
            year_max = st.number_input("Year to", 1900, 2030, 2030, key="exp_yr_max")
        with c4:
            min_rating = st.slider("Min rating", 0.0, 10.0, 0.0, 0.5, key="exp_rating")
            sort_by = st.selectbox("Sort by", ["rating", "year", "title"])

    # Actor search
    actor_filter = st.text_input("Filter by actor name (partial match)", key="exp_actor")

    # Manual text search within DB
    text_filter = st.text_input("Quick title search", placeholder="Type part of a movie title…", key="exp_title")

    # Page size
    page_size = st.select_slider("Movies per page", [10, 20, 30, 50], value=20)

    # Build filters
    filters = {}
    if sel_genre != "Any":
        filters["genre"] = sel_genre
    if sel_lang_name != "Any" and sel_lang:
        filters["language"] = sel_lang
    if year_min > 1900:
        filters["year_min"] = year_min
    if year_max < 2030:
        filters["year_max"] = year_max
    if min_rating > 0:
        filters["min_rating"] = min_rating
    if actor_filter.strip():
        filters["actor"] = actor_filter.strip()

    # Text search takes priority over browse
    if text_filter.strip():
        with st.spinner("Searching…"):
            results = engine.search(text_filter.strip(), top_k=page_size, filters=filters or None)
        render_results_grid(results, title="Search Results", show_score=False)
        return

    # Pagination
    if "explore_page" not in st.session_state:
        st.session_state.explore_page = 1

    # Reset to page 1 when filters change
    filter_key = str(sorted(filters.items())) + sort_by
    if st.session_state.get("last_filter_key") != filter_key:
        st.session_state.explore_page = 1
        st.session_state.last_filter_key = filter_key

    data = engine.browse(
        page=st.session_state.explore_page,
        page_size=page_size,
        filters=filters or None,
        sort_by=sort_by,
    )

    movies = data["movies"]
    total = data["total"]
    total_pages = data["total_pages"]

    st.markdown(f"**{total:,} movies** match your filters &nbsp;|&nbsp; Page {data['page']}/{total_pages}")

    # Pagination controls
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.session_state.explore_page > 1:
            if st.button("← Previous"):
                st.session_state.explore_page -= 1
                st.rerun()
    with c2:
        go_to = st.number_input("Go to page", 1, max(1, total_pages), st.session_state.explore_page, label_visibility="collapsed")
        if go_to != st.session_state.explore_page:
            st.session_state.explore_page = go_to
            st.rerun()
    with c3:
        if st.session_state.explore_page < total_pages:
            if st.button("Next →"):
                st.session_state.explore_page += 1
                st.rerun()

    for idx, movie in enumerate(movies):
        render_movie_card(movie, idx=idx, show_score=False)


# ── Tab 5: Database Manager ──────────────────────────────────────────────────
def tab_database_manager():
    st.markdown("## Database Manager")
    st.markdown('<p style="color:#8888aa">Update, expand, and maintain your movie database.</p>', unsafe_allow_html=True)

    engine = get_search_engine()
    mgr = get_ingestion_manager()
    stats = engine.get_stats()
    meta = mgr.get_metadata()

    # Status card
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movies", f"{stats.get('total_movies', 0):,}")
    with col2:
        last_update = meta.get("last_update", "Never")[:19] if meta else "Never"
        st.metric("Last Updated", last_update)
    with col3:
        version = meta.get("version", 0) if meta else 0
        st.metric("DB Version", f"v{version}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    update_type = st.radio(
        "Update type",
        ["Incremental (last 7 days)", "Targeted (by language / genre / year)", "Full Re-ingest"],
        horizontal=False,
    )

    # ── Incremental ──
    if "Incremental" in update_type:
        days = st.slider("Fetch movies from last N days", 1, 90, 7)
        if st.button("Run Incremental Update", type="primary"):
            with st.spinner(f"Fetching movies from last {days} days…"):
                result = mgr.incremental_update(days=days)
            engine.reload()
            st.success(f"Done! Added **{result['new_movies']}** new movies. Total: {result['total_movies']:,}")

    # ── Targeted ──
    elif "Targeted" in update_type:
        st.markdown("#### Configure targeted fetch")
        c1, c2 = st.columns(2)
        with c1:
            lang_options = {f"{v} ({k})": k for k, v in LANGUAGE_NAMES.items()}
            sel_langs = st.multiselect("Languages to fetch", list(lang_options.keys()))
            chosen_langs = [lang_options[l] for l in sel_langs]

            genre_options = list(GENRE_MAP.values())
            sel_genres = st.multiselect("Genres to fetch", genre_options)

        with c2:
            year_min = st.number_input("Year from", 1900, 2030, 2020, key="db_yr_min")
            year_max = st.number_input("Year to", 1900, 2030, 2026, key="db_yr_max")
            pages = st.slider("Pages per category", 1, 20, 5)

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            if st.button("Run Targeted Update", type="primary"):
                with st.spinner("Fetching targeted movies…"):
                    result = mgr.targeted_update(
                        languages=chosen_langs or None,
                        genres=sel_genres or None,
                        year_min=year_min if year_min > 1900 else None,
                        year_max=year_max if year_max < 2030 else None,
                        pages=pages,
                    )
                engine.reload()
                st.success(f"Done! Added **{result['new_movies']}** new movies. Total: {result['total_movies']:,}")

    # ── Full re-ingest ──
    else:
        st.warning("Warning: Full re-ingest replaces the entire database and may take 10–30 minutes.")
        c1, c2 = st.columns(2)
        with c1:
            fetch_details = st.checkbox("Fetch full movie details (slower, richer data)", value=False)
        with c2:
            include_langs_full = st.multiselect(
                "Extra languages",
                list(LANGUAGE_NAMES.values()),
                default=["Hindi", "Korean", "Japanese", "Tamil"],
                key="full_langs"
            )
        lang_map_rev = {v: k for k, v in LANGUAGE_NAMES.items()}
        chosen_langs_full = [lang_map_rev[l] for l in include_langs_full if l in lang_map_rev]

        confirm = st.checkbox("I understand this will overwrite the existing database")
        if confirm and st.button("Start Full Re-ingest", type="primary"):
            with st.spinner("Running full ingestion… this will take a while"):
                result = mgr.full_ingest(
                    fetch_details=fetch_details,
                    include_languages=chosen_langs_full,
                )
            engine.reload()
            st.success(f"Full ingest complete! {result['total_movies']:,} movies in {result['duration_seconds']:.0f}s")

    # History
    if meta and meta.get("history"):
        with st.expander("Update History"):
            history = meta["history"][-10:][::-1]
            for entry in history:
                ts = entry.get("timestamp", "")[:19]
                op = entry.get("operation", "")
                tot = entry.get("total_movies", 0)
                st.markdown(
                    f'<span class="movie-meta">{ts} — {op} — {tot:,} movies</span>',
                    unsafe_allow_html=True
                )


# ── Tab 6: Statistics ────────────────────────────────────────────────────────
def tab_statistics():
    st.markdown("## Database Statistics")

    engine = get_search_engine()
    stats = engine.get_stats()

    if stats.get("total_movies", 0) == 0:
        st.warning("No movies in database yet.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Movies", f"{stats['total_movies']:,}")
    with c2:
        st.metric("Genres", len(stats.get("genres", {})))
    with c3:
        st.metric("Languages", len(stats.get("languages", {})))
    with c4:
        st.metric("Vector Dim", stats.get("embedding_dimension", 0))

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### Top Genres")
            genres = stats.get("genres", {})
            df_genres = pd.DataFrame(list(genres.items())[:15], columns=["Genre", "Count"])
            fig = px.bar(df_genres, x="Count", y="Genre", orientation="h",
                         color="Count", color_continuous_scale="Reds",
                         template="plotly_dark")
            fig.update_layout(
                plot_bgcolor="#0d0d14", paper_bgcolor="#0d0d14",
                showlegend=False, coloraxis_showscale=False, height=420
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.markdown("#### Movies by Decade")
            decades = stats.get("decades", {})
            df_dec = pd.DataFrame(
                [(f"{k}s", v) for k, v in sorted(decades.items())],
                columns=["Decade", "Count"]
            )
            fig2 = px.bar(df_dec, x="Decade", y="Count",
                          color="Count", color_continuous_scale="Blues",
                          template="plotly_dark")
            fig2.update_layout(
                plot_bgcolor="#0d0d14", paper_bgcolor="#0d0d14",
                showlegend=False, coloraxis_showscale=False, height=420
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Language pie
        st.markdown("#### Language Distribution (Top 15)")
        langs = stats.get("languages", {})
        top_langs = dict(list(sorted(langs.items(), key=lambda x: -x[1]))[:15])
        df_lang = pd.DataFrame(
            [(language_name(k), v) for k, v in top_langs.items()],
            columns=["Language", "Count"]
        )
        fig3 = px.pie(df_lang, names="Language", values="Count",
                      hole=0.45, template="plotly_dark",
                      color_discrete_sequence=px.colors.sequential.RdBu)
        fig3.update_layout(
            plot_bgcolor="#0d0d14", paper_bgcolor="#0d0d14", height=400
        )
        st.plotly_chart(fig3, use_container_width=True)

    except ImportError:
        # Fallback without plotly
        st.markdown("#### Genres")
        for genre, count in list(stats.get("genres", {}).items())[:10]:
            st.markdown(f"**{genre}**: {count}")

        st.markdown("#### Languages")
        for lang, count in list(stats.get("languages", {}).items())[:10]:
            st.markdown(f"**{language_name(lang)}**: {count}")


# ── Main layout ──────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # Hero header
    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0">
      <h1 style="font-size:42px;color:#ffffff!important;margin:0">
        MovieMatch <span style="color:#e50914">AI</span>
      </h1>
      <p style="color:#8888aa;font-size:16px;margin-top:6px">
        Multimodal Movie Discovery — Text · Image · Video · URL
      </p>
    </div>
    """, unsafe_allow_html=True)

    tab_labels = ["Text Search", "Image Search", "Video / URL", "Explore", "Database", "Stats"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_text_search()
    with tabs[1]:
        tab_image_search()
    with tabs[2]:
        tab_video_search()
    with tabs[3]:
        tab_explore()
    with tabs[4]:
        tab_database_manager()
    with tabs[5]:
        tab_statistics()


if __name__ == "__main__":
    main()

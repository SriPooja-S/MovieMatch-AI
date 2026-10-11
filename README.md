# MovieMatch AI - Multimodal Movie Discovery Engine

> **Project Status: Ongoing** - Active development focused on expanding movie catalog coverage, hybrid search ranking, and refining multimodal recognition.

MovieMatch AI is an intelligent movie discovery platform that allows users to find movies using natural language queries, image uploads (posters or scenes), and video clips/URLs (YouTube, Instagram).

---

## How It Works

1. **Ingestion and Enrichment**: Movie metadata (titles, overviews, genres, cast, crew, keywords) is collected from The Movie Database (TMDB) API and compiled into enriched contextual descriptions.
2. **Dense Vector Embeddings**: Movie descriptions are converted into 384-dimensional dense semantic vectors using a SentenceTransformer model (`all-MiniLM-L6-v2`) and L2-normalized.
3. **Multimodal Query Processing**:
   - **Text Query**: Direct semantic embedding generation via Sentence Transformers.
   - **Image Query**: Zero-shot classification with OpenAI CLIP across 70+ curated visual movie prompts, combined with optical character recognition (OCR) via Tesseract for poster text.
   - **Video / URL Query**: Video frame extraction via OpenCV or metadata parsing via yt-dlp, followed by CLIP visual analysis.
4. **Vector Search and Ranking**: Query embeddings are matched against pre-indexed movie vectors using FAISS (or optimized NumPy dot product), delivering ranked match scores (0-100%).
5. **Streaming Availability**: Live streaming and rental provider details are retrieved on demand from TMDB Watch Providers.

---

## Tools and Technologies Used

- **Web Interface**: Streamlit (single-page dashboard with dark theme), Plotly (analytics charts).
- **Embeddings and NLP**: `sentence-transformers` (`all-MiniLM-L6-v2`), PyTorch.
- **Computer Vision**: OpenAI CLIP (`ViT-B/32`), PIL (Pillow), OpenCV (`cv2`).
- **Text Recognition**: Tesseract OCR (`pytesseract`).
- **Vector Search**: FAISS-CPU / NumPy.
- **Video and URL Extraction**: `yt-dlp`.
- **Data Source**: The Movie Database (TMDB) REST API (`requests`, `urllib3`).
- **Data Processing and Storage**: Pandas, NumPy, JSON / NumPy binaries.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Optional: `tesseract-ocr` (`sudo apt install tesseract-ocr`)

### 2. Setup and Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Application
```bash
./run.sh
# or manually:
streamlit run moviematch_app.py
```
Open **http://localhost:8501** in your browser.

---

## Project Structure

```
├── cinematch_app.py        # Streamlit web interface (6 tabs)
├── run.sh                  # Application launch script
├── requirements.txt        # Python dependencies
├── .env                    # Configuration and API keys
└── app/
    ├── core/               # Search engine, embedder, CLIP processor, vector store
    ├── ingestion/          # TMDB client, data enrichment, ingestion manager
    ├── processors/         # Image (OCR), Video (OpenCV), and URL (yt-dlp) processors
    ├── utils/              # Configuration and logging
    └── data/               # Movie database (JSON), embeddings (NumPy), metadata
```

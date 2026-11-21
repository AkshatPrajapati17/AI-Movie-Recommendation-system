# CineMatch AI — Intelligent Movie Recommendation System 🎬

A modern, Python-only college project for movie recommendation and exploratory data analysis powered by the **MovieLens** dataset, **Streamlit**, and authentic **TMDb movie posters**.

---

## 📌 Project Overview

**CineMatch AI** is a complete, multi-signal movie recommendation engine and discovery platform:
- **Offline-First Architecture**: App runs 100% locally on Windows without requiring network calls during Streamlit reruns.
- **Genuine Movie Posters**: Fetches and verifies real posters from the official TMDb API via `fetch_posters.py`, mapping MovieLens films through `data/links.csv` and title/year matching.
- **No Fabricated Data**: Authentic ratings, real community tags, and strict schema validation. Never invents image URLs or assigns posters to unverified films.
- **Five Specialized Recommendation Engines**:
  1. **Discover**: Rich browsable catalog with genre filtering, popularity sorting, and responsive cinema cards.
  2. **Similar Movies (Content-Based)**: TF-IDF & Cosine Similarity over genres and community tags.
  3. **Collaborative Recommendations**: Item-Item sparse rating matrix affinity across 610 MovieLens profiles.
  4. **Semantic Search**: Neural embedding vectors (`all-MiniLM-L6-v2`) with cosine distance ranking for natural-language mood and theme prompts.
  5. **Hybrid Engine**: Dynamic weighted ensemble combining Content, Collaborative, and Semantic signals.
- **Dark Cinematic Theme**: Deep charcoal background (`#101116`), burgundy velvet accents (`#6E2338`), warm gold highlights (`#E6B566`), and ivory text (`#F3EBDD`).
- **Resilient UI**: Uses native `st.image()` for verified local posters and falls back to polished typographic cinema cards if a poster is absent or pending download.

---

## 📁 Project Architecture

```text
CineMatch-AI/
├── app.py                  # Streamlit web application & cinematic discovery UI
├── poster_service.py       # TMDb poster matching, downloading, and manifest service
├── fetch_posters.py        # CLI batch poster ingestion and resume utility
├── data_loader.py          # Data ingestion, schema validation, and statistical profiling
├── content_based.py        # TF-IDF & Cosine Similarity content recommender
├── collaborative.py        # Item-Item & User rating matrix collaborative filtering
├── semantic_search.py      # Neural NLP semantic search & vector cache
├── hybrid_recommender.py   # Tri-model hybrid ensemble recommender
├── requirements.txt        # Project dependencies (streamlit, pandas, scikit-learn, etc.)
├── .env.example            # Template for TMDb API credentials
├── .gitignore              # Ignores .env, virtualenv, and local poster media
├── README.md               # Setup, API key configuration, and documentation
├── data/                   # MovieLens ml-latest-small dataset
│   ├── movies.csv          # [Required] movieId, title, genres
│   ├── ratings.csv         # [Required] userId, movieId, rating, timestamp
│   ├── tags.csv            # [Optional] userId, movieId, tag, timestamp
│   ├── links.csv           # [Optional] movieId, imdbId, tmdbId (for poster lookup)
│   └── README.txt          # MovieLens dataset documentation
├── posters/                # Local verified movie poster directory
│   ├── manifest.json       # JSON manifest mapping movieId to match status & paths
│   └── {movieId}.jpg       # Downloaded authentic poster images
├── cache/                  # Persistent embedding vectors and index markers
└── tests/                  # Automated test suite (59 unit tests)
    ├── test_data_loader.py
    ├── test_content_based.py
    ├── test_collaborative.py
    ├── test_semantic_search.py
    ├── test_hybrid_recommender.py
    └── test_poster_service.py
```

---

## 🖼️ TMDb Poster Pipeline & Configuration

### 1. How to Obtain a Free TMDb API Key
1. Create a free account at [themoviedb.org/signup](https://www.themoviedb.org/signup).
2. Verify your email address and log in.
3. Navigate to **Account Settings > API** ([themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)).
4. Click **Create** / **Request an API Key** (choose Developer / Educational for college projects).
5. Copy your **API Key (v3 auth)**.

### 2. Configure Your `.env` File
Create a `.env` file in the project root (you can copy `.env.example`):
```bash
cp .env.example .env
```
Add your key:
```ini
TMDB_API_KEY=your_actual_tmdb_api_key_here
```
> **Security Note**: `.env` is included in `.gitignore` and is never exposed in the UI, logs, or committed to version control.

### 3. Check Current Poster Manifest Status
Inspect current download and match statistics anytime:
```powershell
python fetch_posters.py --status
```

### 4. Test on a Small Batch First
Before processing the full 9,742 movies, test with a small batch (e.g., 10 films):
```powershell
python fetch_posters.py --limit 10
```
This fetches verified posters for the first 10 MovieLens titles, downloads the image files into `posters/{movieId}.jpg`, and updates `posters/manifest.json`.

### 5. Fetch Posters for the Full Dataset
To run the full ingestion pipeline:
```powershell
python fetch_posters.py
```
- **Resumable**: If cancelled (`Ctrl+C`) or interrupted, the process safely saves the manifest and resumes where it left off on the next run without re-downloading existing posters.
- **Polite Rate Limiting**: Defaults to 0.25s delay between requests to comply with TMDb guidelines and prevent HTTP 429 rate limits.
- **Multi-Tier Matching**: Matches via `tmdbId` (Tier 1), `imdbId` (Tier 2), and clean title + year search (Tier 3-5).

---

## ⚙️ Quickstart Guide for Windows

### Prerequisites
- Windows 10 or 11
- Python 3.10+ (accessible via `python` or `py`)

---

### Step 1: Open PowerShell
Navigate to the project root directory:
```powershell
cd c:\Users\RD\Desktop\movie
```

---

### Step 2: Virtual Environment Setup
Activate the included virtual environment:
```powershell
# In PowerShell:
.\.venv\Scripts\Activate.ps1
```
*(Or in Command Prompt: `.\.venv\Scripts\activate.bat`)*

---

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

### Step 4: Run Automated Unit Tests
Verify all 59 tests pass across data loading, recommenders, and the poster pipeline:
```powershell
python -m unittest discover -s tests
```

---

### Step 5: Launch CineMatch AI
Launch the Streamlit application:
```powershell
streamlit run app.py
```
The app will open automatically in your default browser at:
👉 **`http://localhost:8501`**

---

## 🎨 Streamlit UI & Movie Card Integration

- **Native `st.image()` Rendering**: Local poster files are displayed using Streamlit's native `st.image()` container with consistent 2:3 cinematic aspect ratios (`w342`), eliminating raw HTML rendering glitches.
- **Polished Typographic Fallback**: For movies without downloaded posters, the UI displays a dark burgundy/gold typographic cinema card with the movie title in Bebas Neue, release year tag, primary genre badge, and star rating.
- **Unified Across All Tabs**: The shared card layout is seamlessly integrated across:
  - Discover Catalog
  - Similar Thematic Movies
  - Collaborative User Recommendations
  - Semantic NLP Search
  - Tri-Model Hybrid Ensemble
- **Attribution & Compliance**: The app footer and sidebar display the required TMDb attribution notice:
  > *This product uses the TMDb API but is not endorsed or certified by TMDb.*

---

## 🛡️ Reliability & Zero-Fabrication Guarantees

| Metric / Rule | Implementation Guarantee |
| :--- | :--- |
| **Offline Independence** | Streamlit reruns perform zero external network requests. Everything is resolved locally from `posters/`. |
| **Recommendation Independence** | Algorithms operate purely on MovieLens data. Poster presence or absence never impacts similarity scores. |
| **No Mismatches** | Multi-tier matching verifies TMDb ID, IMDb ID, clean title, and release year before saving. |
| **No Invented URLs** | If TMDb has no poster or the film is unverified, `manifest.json` marks it as `no_poster` or `not_found`. |
| **Safe Storage** | Posters use sanitized numeric filenames (`posters/{movieId}.jpg`). |

---

## 📊 Dataset Profile (MovieLens ml-latest-small)

- **Total Movies**: 9,742
- **Total Ratings**: 100,836
- **Unique Users**: 610
- **Average Rating**: 3.50 ⭐ (scale: 0.5 – 5.0)
- **Unique Genres**: 19
- **User Tags**: 3,683 verified community tags
- **External Links**: 9,742 IMDb / TMDb identifiers

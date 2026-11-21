"""
semantic_search.py
==================
CineMatch AI — Intelligent Movie Recommendation System
NLP-Based Semantic Movie Search Module using sentence-transformers (all-MiniLM-L6-v2)
with local embedding caching, cache validation, and graceful TF-IDF fallback.

Design & Compliance:
1. Embeddings generated from authentic text (title, genres, community tags).
   No fabricated plot synopses or synthetic summaries.
2. Uses all-MiniLM-L6-v2 sentence embedding model (an embedding model, not an LLM).
3. Supports natural-language semantic queries:
   - "science fiction movies about space"
   - "funny movies for a family night"
   - "dark crime thrillers"
   - "romantic movies with a sad ending"
4. Compares search query embedding with movie embeddings via cosine similarity.
5. Persistent disk caching in `cache/` directory with automatic MD5 hash validation
   to invalidate stale caches if the dataset changes.
6. Graceful degradation: If sentence-transformers or the model fails to load/download,
   automatically falls back to a TF-IDF keyword search and notifies the user.
7. 100% local, no API keys or external inference services required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional Streamlit caching helper
try:
    import streamlit as st
    cache_resource_decorator = st.cache_resource
except Exception:  # pragma: no cover
    def cache_resource_decorator(*args, **kwargs):
        def inner(func):
            return func
        return inner


class MovieSemanticSearch:
    """
    NLP Semantic Search Engine for CineMatch AI.

    Encodes authentic movie metadata into vector embeddings and matches natural language
    queries using cosine similarity. Seamlessly falls back to TF-IDF when neural
    embeddings are unavailable.
    """

    DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None,
        cache_dir: str = "cache",
        model_name: str = DEFAULT_MODEL_NAME,
        force_fallback: bool = False,
    ) -> None:
        """
        Initializes the semantic search engine.

        Args:
            movies_df: Validated DataFrame of movies from movies.csv.
            tags_df: Optional validated DataFrame of tags from tags.csv.
            cache_dir: Directory to save and read local cached embeddings.
            model_name: HuggingFace sentence-transformers model identifier.
            force_fallback: When True, bypasses neural model and uses TF-IDF fallback (for testing).
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.tags_df = tags_df
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.force_fallback = force_fallback

        # Engine state flags
        self.is_fallback: bool = False
        self.engine_type: str = f"Neural Semantic Embeddings ({self.model_name})"
        self.fallback_reason: Optional[str] = None

        # Data structures
        self.corpus: List[str] = []
        self.movie_ids: List[int] = self.movies_df["movieId"].astype(int).tolist()
        self.embeddings: Optional[np.ndarray] = None
        self.vectorizer: Any = None
        self.tfidf_matrix: Any = None
        self.model: Any = None

        # Prepare corpus and initialize search representation
        self._build_corpus()
        self._initialize_engine()

    def _build_corpus(self) -> None:
        """
        Constructs rich, natural textual descriptions per movie from authentic fields.
        Omits plot summaries and directors as they are not present in the local CSV dataset.
        """
        # Aggregate tags per movie
        tags_by_movie: Dict[int, str] = {}
        if self.tags_df is not None and not self.tags_df.empty:
            for mid, group in self.tags_df.groupby("movieId"):
                clean_tags = [
                    re.sub(r"[^\w\s-]", "", str(t).strip())
                    for t in group["tag"].dropna().unique()
                    if str(t).strip()
                ]
                tags_by_movie[int(mid)] = ", ".join(clean_tags[:8])

        corpus_docs = []
        for _, row in self.movies_df.iterrows():
            mid = int(row["movieId"])
            title = str(row["title"]).strip()
            year_val = row.get("year")
            year_str = f"({year_val})" if pd.notna(year_val) and year_val else ""

            raw_genres = str(row["genres"]).strip()
            if raw_genres and raw_genres != "(no genres listed)":
                genres_clean = ", ".join([g.strip() for g in raw_genres.split("|") if g.strip()])
            else:
                genres_clean = "Uncategorized"

            movie_tags = tags_by_movie.get(mid, "")

            # Combine fields into an expressive natural language prompt
            doc_parts = [f"Movie: {title} {year_str}", f"Genres: {genres_clean}"]
            if movie_tags:
                doc_parts.append(f"Themes and community tags: {movie_tags}")

            doc = ". ".join(doc_parts) + "."
            corpus_docs.append(doc)

        self.corpus = corpus_docs

    def _compute_corpus_hash(self) -> str:
        """Computes MD5 hash of the corpus and metadata to detect dataset changes."""
        hasher = hashlib.md5()
        hasher.update(str(len(self.corpus)).encode("utf-8"))
        hasher.update(self.model_name.encode("utf-8"))
        # Sample slices of corpus for fast robust checksum
        step = max(1, len(self.corpus) // 100)
        for i in range(0, len(self.corpus), step):
            hasher.update(self.corpus[i].encode("utf-8"))
        return hasher.hexdigest()

    def _initialize_engine(self) -> None:
        """Initializes the sentence embedding model or activates TF-IDF fallback."""
        if self.force_fallback:
            self._activate_fallback("Fallback explicitly forced by configuration.")
            return

        try:
            from sentence_transformers import SentenceTransformer

            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, "semantic_embeddings.npz")
            meta_file = os.path.join(self.cache_dir, "semantic_embeddings_meta.json")

            current_hash = self._compute_corpus_hash()
            cache_valid = False

            # Check if valid cached embeddings exist on disk
            if os.path.isfile(cache_file) and os.path.isfile(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if (
                        meta.get("corpus_hash") == current_hash
                        and meta.get("num_movies") == len(self.movies_df)
                        and meta.get("model_name") == self.model_name
                    ):
                        cached_data = np.load(cache_file)
                        self.embeddings = cached_data["embeddings"]
                        cache_valid = True
                        logger.info("Loaded cached embeddings from disk for %d movies.", len(self.embeddings))
                except Exception as cache_err:
                    logger.warning("Cache validation error: %s. Recomputing embeddings.", cache_err)

            # Load model (downloads once if not cached locally by HuggingFace)
            logger.info("Loading sentence-transformers model '%s'...", self.model_name)
            self.model = SentenceTransformer(self.model_name)

            if not cache_valid or self.embeddings is None:
                logger.info("Generating embeddings for %d movies with %s...", len(self.corpus), self.model_name)
                # Compute normalized embeddings in batches for high speed
                self.embeddings = self.model.encode(
                    self.corpus,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                # Persist embeddings and verification metadata
                try:
                    np.savez_compressed(cache_file, embeddings=self.embeddings)
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "corpus_hash": current_hash,
                                "num_movies": len(self.movies_df),
                                "model_name": self.model_name,
                            },
                            f,
                            indent=2,
                        )
                    logger.info("Successfully persisted semantic embeddings to %s", cache_file)
                except Exception as save_err:
                    logger.warning("Failed to write embedding cache to disk: %s", save_err)

            self.is_fallback = False
            self.engine_type = f"Neural Semantic Embeddings ({self.model_name})"

        except Exception as exc:
            logger.warning("sentence-transformers model could not be initialized (%s). Activating TF-IDF fallback.", exc)
            self._activate_fallback(str(exc))

    def _activate_fallback(self, reason: str) -> None:
        """Activates TF-IDF keyword search fallback when neural embeddings cannot be loaded."""
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

        self.is_fallback = True
        self.engine_type = "TF-IDF Keyword Search (Fallback)"
        self.fallback_reason = reason

        custom_stopwords = list(ENGLISH_STOP_WORDS.union({"movie", "movies", "film", "films"}))
        self.vectorizer = TfidfVectorizer(
            stop_words=custom_stopwords,
            min_df=1,
            token_pattern=r"(?u)\b[\w-]+\b",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info("TF-IDF fallback search initialized on %d documents.", len(self.corpus))

    def search(self, query: str, top_n: int = 15) -> pd.DataFrame:
        """
        Executes a natural language semantic search across the movie catalog.

        Args:
            query: Natural language query string (e.g. 'science fiction movies about space').
            top_n: Maximum matching movies to return.

        Returns:
            pd.DataFrame: Matched movies ranked by cosine similarity score.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "year",
                    "similarity_score",
                    "similarity_percentage",
                ]
            )

        if not self.is_fallback and self.model is not None and self.embeddings is not None:
            # Neural sentence-transformer cosine similarity
            q_emb = self.model.encode(cleaned_query, normalize_embeddings=True, convert_to_numpy=True)
            # Dot product with pre-normalized vectors equals cosine similarity
            scores = np.dot(self.embeddings, q_emb).flatten()
        else:
            # TF-IDF fallback cosine similarity
            from sklearn.metrics.pairwise import linear_kernel

            q_vec = self.vectorizer.transform([cleaned_query])
            scores = linear_kernel(q_vec, self.tfidf_matrix).flatten()

        # Rank indices in descending order
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0 and len(results) >= min(top_n, 3):
                break

            movie_row = self.movies_df.iloc[idx]
            results.append({
                "movieId": int(movie_row["movieId"]),
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "year": movie_row.get("year"),
                "similarity_score": round(score, 4),
                "similarity_percentage": f"{max(0.0, score) * 100:.1f}%",
            })

            if len(results) >= top_n:
                break

        res_df = pd.DataFrame(results)
        if res_df.empty:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "year",
                    "similarity_score",
                    "similarity_percentage",
                ]
            )

        return res_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streamlit-Cached Search Factory
# ---------------------------------------------------------------------------

@cache_resource_decorator(show_spinner=False)
def get_cached_semantic_search(
    movies_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
    cache_dir: str = "cache",
    model_name: str = MovieSemanticSearch.DEFAULT_MODEL_NAME,
) -> MovieSemanticSearch:
    """
    Creates and caches a MovieSemanticSearch engine across Streamlit interactions.

    Args:
        movies_df: Validated DataFrame of movies.
        tags_df: Optional validated DataFrame of tags.
        cache_dir: Local cache folder path.
        model_name: Embedding model identifier.

    Returns:
        MovieSemanticSearch: Loaded search engine.
    """
    return MovieSemanticSearch(
        movies_df=movies_df,
        tags_df=tags_df,
        cache_dir=cache_dir,
        model_name=model_name,
    )

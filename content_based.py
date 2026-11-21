"""
content_based.py
================
CineMatch AI — Intelligent Movie Recommendation System
Content-Based Filtering Module using scikit-learn TfidfVectorizer & Cosine Similarity.

Design & Compliance:
1. Uses scikit-learn's `TfidfVectorizer` and `cosine_similarity` / `linear_kernel`.
2. Text representation built exclusively from verified local fields (genres and authentic tags).
   - Plot descriptions, directors, and cast are not included because they do not exist
     in the local MovieLens CSV files. We never invent or fabricate fake metadata.
3. Normalizes missing values, handles '(no genres listed)', and merges community tags safely.
4. High-performance sparse vector dot-product calculation for instantaneous recommendations.
5. Comprehensive edge case handling: unknown movie IDs, movies with no usable features,
   and zero-norm vectors.
6. Cached model builder for Streamlit with graceful fallback.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

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


class ContentBasedRecommender:
    """
    Content-Based Movie Recommendation Engine.
    
    Computes pairwise content similarity across movies using TF-IDF feature matrices
    derived from authentic MovieLens genres and verified community tags.
    """

    def __init__(
        self,
        movies_df: pd.DataFrame,
        tags_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initializes the content-based recommender.

        Args:
            movies_df: Validated DataFrame of movies from movies.csv.
            tags_df: Optional validated DataFrame of user tags from tags.csv.
        """
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.tags_df = tags_df

        # Fast lookup indices
        self._movie_id_to_idx: Dict[int, int] = {
            int(mid): idx for idx, mid in enumerate(self.movies_df["movieId"])
        }
        self._idx_to_movie_id: Dict[int, int] = {
            idx: int(mid) for idx, mid in enumerate(self.movies_df["movieId"])
        }

        self.vectorizer = None
        self.tfidf_matrix = None
        self.feature_names: List[str] = []
        self._metadata_soup: Optional[pd.Series] = None
        self._is_fitted = False

    def build_metadata_soup(self) -> pd.Series:
        """
        Builds a normalized text representation from available fields (genres and tags).

        Note: Plot descriptions, directors, and actors are omitted because they do not
        exist in the local MovieLens dataset files. No synthetic text is fabricated.

        Returns:
            pd.Series: Combined textual document per movie.
        """
        # 1. Clean and normalize genres
        def clean_genres(genres_str: Union[str, float]) -> str:
            if not isinstance(genres_str, str) or genres_str.strip() == "" or genres_str == "(no genres listed)":
                return ""
            # Split pipe and preserve compound terms (e.g. 'Sci-Fi' -> 'SciFi sci-fi')
            tokens = []
            for g in genres_str.split("|"):
                g_clean = g.strip()
                if g_clean and g_clean != "(no genres listed)":
                    tokens.append(g_clean)
                    tokens.append(g_clean.replace("-", ""))
            # Repeat genre tokens twice to give strong weight to primary genre categories
            return " ".join(tokens * 2)

        genres_text = self.movies_df["genres"].apply(clean_genres)

        # 2. Aggregate authentic community tags if provided
        if self.tags_df is not None and not self.tags_df.empty:
            # Group by movieId and clean tags
            tag_agg = (
                self.tags_df.dropna(subset=["tag"])
                .groupby("movieId")["tag"]
                .apply(lambda tags: " ".join({
                    re.sub(r"[^\w\s-]", "", str(t).lower().strip())
                    for t in tags if str(t).strip()
                }))
                .reset_index()
            )
            merged = self.movies_df[["movieId"]].merge(tag_agg, on="movieId", how="left")
            tags_text = merged["tag"].fillna("").astype(str)
        else:
            tags_text = pd.Series([""] * len(self.movies_df), index=self.movies_df.index)

        # 3. Combine normalized genres and tags
        combined = (genres_text + " " + tags_text).str.strip().str.lower()
        self._metadata_soup = combined
        return combined

    def fit(self) -> "ContentBasedRecommender":
        """
        Fits the scikit-learn TfidfVectorizer on the combined movie metadata soup.

        Returns:
            ContentBasedRecommender: The fitted recommender instance.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        corpus = self.build_metadata_soup().tolist()

        # Build TF-IDF with English stop-words and token pattern that preserves hyphens
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b[\w-]+\b",
            min_df=1,
            sublinear_tf=True,
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        try:
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        except AttributeError:  # pragma: no cover
            self.feature_names = self.vectorizer.get_feature_names()

        self._is_fitted = True
        logger.info(
            "Fitted ContentBasedRecommender on %d movies with %d TF-IDF features.",
            self.tfidf_matrix.shape[0],
            self.tfidf_matrix.shape[1],
        )
        return self

    def has_usable_features(self, movie_id: int) -> bool:
        """
        Checks whether a given movie ID has non-empty metadata features.

        Args:
            movie_id: Target movie ID.

        Returns:
            bool: True if movie has non-zero TF-IDF features.
        """
        if not self._is_fitted or self.tfidf_matrix is None:
            raise ValueError("Recommender must be fitted before checking features.")

        if movie_id not in self._movie_id_to_idx:
            return False

        idx = self._movie_id_to_idx[movie_id]
        row_vec = self.tfidf_matrix[idx]
        return row_vec.nnz > 0

    def recommend(
        self,
        movie_id: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommends top-N movies most similar to the selected movie ID.

        Excludes the selected movie itself. Uses efficient sparse vector dot product
        (linear_kernel) for cosine similarity computation.

        Args:
            movie_id: Target MovieLens movie ID.
            top_n: Number of recommendations to return (default: 10).

        Returns:
            pd.DataFrame: Recommended movies containing:
                - movieId
                - title
                - genres
                - year
                - similarity_score (float in [0.0, 1.0])
                - similarity_percentage (string, e.g. '87.5%')

        Raises:
            ValueError: If recommender is not fitted or movie_id does not exist.
        """
        if not self._is_fitted or self.tfidf_matrix is None:
            raise ValueError("Recommender model must be fitted with fit() before calling recommend().")

        if movie_id not in self._movie_id_to_idx:
            raise ValueError(f"Movie ID {movie_id} was not found in the movie catalog.")

        target_idx = self._movie_id_to_idx[movie_id]
        target_vec = self.tfidf_matrix[target_idx]

        # Check for movies with no usable features
        if target_vec.nnz == 0:
            logger.warning(
                "Movie ID %d has no usable genres or tags for content recommendation.",
                movie_id,
            )
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

        # Efficient sparse cosine similarity using linear_kernel
        from sklearn.metrics.pairwise import linear_kernel

        cosine_sims = linear_kernel(target_vec, self.tfidf_matrix).flatten()

        # Rank indices in descending order of similarity
        ranked_indices = np.argsort(cosine_sims)[::-1]

        # Exclude selected movie itself and select top_n with non-zero similarity
        results = []
        for idx in ranked_indices:
            if idx == target_idx:
                continue
            score = float(cosine_sims[idx])
            if score <= 0.0 and len(results) >= min(top_n, 5):
                # Stop if similarity is zero and we have minimum matches
                break

            mid = self._idx_to_movie_id[idx]
            movie_row = self.movies_df.iloc[idx]
            results.append({
                "movieId": mid,
                "title": movie_row["title"],
                "genres": movie_row["genres"],
                "year": movie_row.get("year"),
                "similarity_score": round(score, 4),
                "similarity_percentage": f"{score * 100:.1f}%",
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
# Streamlit-Cached Recommender Factory
# ---------------------------------------------------------------------------

@cache_resource_decorator(show_spinner=False)
def get_cached_content_recommender(
    movies_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
) -> ContentBasedRecommender:
    """
    Creates, fits, and caches a ContentBasedRecommender instance.
    Cached across user interactions in Streamlit.

    Args:
        movies_df: Validated DataFrame of movies.
        tags_df: Optional validated DataFrame of tags.

    Returns:
        ContentBasedRecommender: Fitted recommender ready for fast queries.
    """
    recommender = ContentBasedRecommender(movies_df=movies_df, tags_df=tags_df)
    recommender.fit()
    return recommender

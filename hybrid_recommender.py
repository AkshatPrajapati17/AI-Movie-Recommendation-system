"""
hybrid_recommender.py
=====================
CineMatch AI — Intelligent Movie Recommendation System
Tri-Model Hybrid Recommendation Engine combining Content-Based Filtering,
Collaborative Filtering, and NLP Semantic Search.

Design & Compliance:
1. Harmonizes three distinct recommendation signals:
   - Content-Based: genre and tag feature similarity from content_based.py.
   - Collaborative: real user rating behavior and co-rating patterns from collaborative.py.
   - Semantic: natural-language query alignment from semantic_search.py (when provided).
2. Dynamic Weight Re-normalization:
   Adjusts weights automatically based on active/available signals so no single
   technique dominates when another is absent (e.g. cold start, missing user, or empty query).
3. Fair Score Normalization:
   Normalizes heterogeneous raw scores (cosine similarities, collaborative dot-products)
   to [0.0, 1.0] before weighted aggregation.
4. Strict De-duplication and Exclusions:
   - Always excludes the selected seed movie.
   - Strictly excludes films already rated by the selected user when a user ID is supplied.
5. Truthful, Transparent Explanations:
   Generates honest explanations based strictly on the signals that actually contributed.
   Never claims user preference understanding without real rating history.
6. Returns both aggregated hybrid scores and granular component scores for transparency.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from semantic_search import MovieSemanticSearch

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


DEFAULT_HYBRID_WEIGHTS: Dict[str, float] = {
    "content": 0.40,
    "collab": 0.35,
    "semantic": 0.25,
}


class HybridRecommender:
    """
    Intelligent Tri-Model Hybrid Recommender for CineMatch AI.
    
    Blends thematic content matching, collective collaborative intelligence,
    and NLP natural-language semantic matching into a unified ranking system.
    """

    def __init__(
        self,
        content_model: ContentBasedRecommender,
        collab_model: CollaborativeRecommender,
        semantic_model: MovieSemanticSearch,
        movies_df: pd.DataFrame,
        ratings_df: Optional[pd.DataFrame] = None,
        default_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initializes the hybrid recommender with fitted component engines.

        Args:
            content_model: Fitted ContentBasedRecommender instance.
            collab_model: Fitted CollaborativeRecommender instance.
            semantic_model: Initialized MovieSemanticSearch instance.
            movies_df: Validated DataFrame of movies.
            ratings_df: Optional validated DataFrame of ratings (for user history checks).
            default_weights: Optional custom base weights for {"content", "collab", "semantic"}.
        """
        self.content_model = content_model
        self.collab_model = collab_model
        self.semantic_model = semantic_model
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df

        self.weights = dict(default_weights or DEFAULT_HYBRID_WEIGHTS)

        # Build fast movie lookup dictionary
        self._movie_meta: Dict[int, Dict[str, Any]] = {}
        for _, row in self.movies_df.iterrows():
            self._movie_meta[int(row["movieId"])] = {
                "title": row["title"],
                "genres": row["genres"],
                "year": row.get("year"),
            }

    def recommend(
        self,
        movie_id: int,
        user_id: Optional[int] = None,
        query: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Generates hybrid recommendations by combining active recommendation signals.

        Args:
            movie_id: Selected seed movie ID (required).
            user_id: Optional MovieLens user ID to personalize recommendations.
            query: Optional natural-language mood or preference query (e.g. 'space adventure').
            weights: Optional custom dictionary of weights {'content': w_c, 'collab': w_col, 'semantic': w_s}.
            top_n: Number of recommendations to return.

        Returns:
            pd.DataFrame: Ranked recommendations with combined score, component scores,
                          and truthful explanations.

        Raises:
            ValueError: If movie_id does not exist in the movie catalog.
        """
        if movie_id not in self._movie_meta:
            raise ValueError(f"Movie ID {movie_id} was not found in the movie catalog.")

        # Candidate pool size (fetch 3x top_n from each active source for thorough overlap)
        pool_size = max(30, top_n * 4)

        # -------------------------------------------------------------------
        # 1. Collect Content-Based Signal
        # -------------------------------------------------------------------
        content_scores: Dict[int, float] = {}
        try:
            c_df = self.content_model.recommend(movie_id=movie_id, top_n=pool_size)
            if not c_df.empty:
                max_c = float(c_df["similarity_score"].max()) or 1.0
                for _, row in c_df.iterrows():
                    mid = int(row["movieId"])
                    raw_s = float(row["similarity_score"])
                    content_scores[mid] = max(0.0, raw_s / max_c)
        except Exception as exc:
            logger.warning("Content model could not generate candidates: %s", exc)

        # -------------------------------------------------------------------
        # 2. Collect Collaborative Signal
        # -------------------------------------------------------------------
        collab_scores: Dict[int, float] = {}
        collab_is_user_personalized = False

        if user_id is not None:
            # Check if user exists in ratings
            try:
                u_df = self.collab_model.recommend_for_user(user_id=user_id, top_n=pool_size)
                if not u_df.empty:
                    collab_is_user_personalized = True
                    max_col = float(u_df["collaborative_score"].max()) or 1.0
                    for _, row in u_df.iterrows():
                        mid = int(row["movieId"])
                        raw_s = float(row["collaborative_score"])
                        collab_scores[mid] = max(0.0, raw_s / max_col)
            except Exception as exc:
                logger.info("Collaborative user lookup failed (%s). Falling back to item similarity.", exc)

        # If user collaborative was not available or no user provided, use item-item collaborative similarity
        if not collab_scores:
            try:
                item_collab_df = self.collab_model.get_similar_movies(movie_id=movie_id, top_n=pool_size)
                if not item_collab_df.empty:
                    max_col = float(item_collab_df["similarity_score"].max()) or 1.0
                    for _, row in item_collab_df.iterrows():
                        mid = int(row["movieId"])
                        raw_s = float(row["similarity_score"])
                        collab_scores[mid] = max(0.0, raw_s / max_col)
            except Exception as exc:
                logger.info("Item collaborative similarity unavailable for movie %d: %s", movie_id, exc)

        # -------------------------------------------------------------------
        # 3. Collect Semantic Search Signal (if query is provided)
        # -------------------------------------------------------------------
        semantic_scores: Dict[int, float] = {}
        has_query = bool(query and query.strip())

        if has_query:
            try:
                sem_df = self.semantic_model.search(query=query.strip(), top_n=pool_size)
                if not sem_df.empty:
                    max_sem = float(sem_df["similarity_score"].max()) or 1.0
                    for _, row in sem_df.iterrows():
                        mid = int(row["movieId"])
                        raw_s = float(row["similarity_score"])
                        semantic_scores[mid] = max(0.0, raw_s / max_sem)
            except Exception as exc:
                logger.warning("Semantic search error: %s", exc)

        # -------------------------------------------------------------------
        # 4. Resolve Active Weights and Dynamic Re-normalization
        # -------------------------------------------------------------------
        cfg_weights = dict(weights or self.weights)
        active_weights: Dict[str, float] = {}

        if content_scores:
            active_weights["content"] = float(cfg_weights.get("content", 0.40))
        if collab_scores:
            active_weights["collab"] = float(cfg_weights.get("collab", 0.35))
        if semantic_scores and has_query:
            active_weights["semantic"] = float(cfg_weights.get("semantic", 0.25))

        # Handle complete cold start (no candidates found from any model)
        if not active_weights:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "year",
                    "hybrid_score",
                    "hybrid_percentage",
                    "content_score",
                    "collab_score",
                    "semantic_score",
                    "explanation",
                    "active_signals",
                ]
            )

        # Re-normalize weights so their sum equals exactly 1.0
        total_weight = sum(active_weights.values())
        norm_weights = {k: v / total_weight for k, v in active_weights.items()}

        # -------------------------------------------------------------------
        # 5. Determine Exclusions (Seed Movie & Already-Rated Movies)
        # -------------------------------------------------------------------
        excluded_movie_ids: Set[int] = {movie_id}

        if user_id is not None and self.ratings_df is not None:
            user_ratings = self.ratings_df[self.ratings_df["userId"] == user_id]
            if not user_ratings.empty:
                excluded_movie_ids.update(user_ratings["movieId"].astype(int))

        # -------------------------------------------------------------------
        # 6. Aggregate Scores across All Candidates
        # -------------------------------------------------------------------
        all_candidate_ids = (set(content_scores.keys()) | set(collab_scores.keys()) | set(semantic_scores.keys()))
        all_candidate_ids -= excluded_movie_ids

        scored_records = []
        seed_meta = self._movie_meta.get(movie_id, {})
        seed_title = seed_meta.get("title", f"Movie #{movie_id}")

        for mid in all_candidate_ids:
            c_score = content_scores.get(mid, 0.0)
            col_score = collab_scores.get(mid, 0.0)
            sem_score = semantic_scores.get(mid, 0.0)

            # Calculate weighted composite score
            final_score = (
                norm_weights.get("content", 0.0) * c_score
                + norm_weights.get("collab", 0.0) * col_score
                + norm_weights.get("semantic", 0.0) * sem_score
            )

            # Generate truthful explanation based strictly on participating signals
            exp_parts = []
            active_signals = []

            if c_score > 0.05 and "content" in norm_weights:
                active_signals.append("Content")
                exp_parts.append(f"shares thematic/genre DNA with {seed_title} ({int(c_score * 100)}% match)")

            if col_score > 0.05 and "collab" in norm_weights:
                active_signals.append("Collaborative")
                if collab_is_user_personalized:
                    exp_parts.append(f"aligns with User #{user_id}'s viewing patterns ({int(col_score * 100)}% affinity)")
                else:
                    exp_parts.append(f"frequently co-rated with {seed_title} by MovieLens viewers ({int(col_score * 100)}% co-rating match)")

            if sem_score > 0.05 and "semantic" in norm_weights and has_query:
                active_signals.append("Semantic")
                clean_q_snippet = query.strip()[:30] + ("..." if len(query.strip()) > 30 else "")
                exp_parts.append(f"matches mood '{clean_q_snippet}' ({int(sem_score * 100)}% alignment)")

            if not exp_parts:
                exp_text = f"Recommended by CineMatch multi-signal ensemble ranking."
            else:
                exp_text = "Recommended because it " + ", and ".join(exp_parts) + "."

            meta = self._movie_meta.get(mid, {})
            scored_records.append({
                "movieId": mid,
                "title": meta.get("title", f"Movie #{mid}"),
                "genres": meta.get("genres", ""),
                "year": meta.get("year"),
                "hybrid_score": round(float(final_score), 4),
                "hybrid_percentage": f"{final_score * 100:.1f}%",
                "content_score": round(float(c_score), 3),
                "collab_score": round(float(col_score), 3),
                "semantic_score": round(float(sem_score), 3),
                "explanation": exp_text,
                "active_signals": active_signals,
            })

        if not scored_records:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "year",
                    "hybrid_score",
                    "hybrid_percentage",
                    "content_score",
                    "collab_score",
                    "semantic_score",
                    "explanation",
                    "active_signals",
                ]
            )

        res_df = pd.DataFrame(scored_records)
        res_df = res_df.sort_values(by="hybrid_score", ascending=False).head(top_n)
        return res_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streamlit-Cached Hybrid Recommender Factory
# ---------------------------------------------------------------------------

@cache_resource_decorator(show_spinner=False)
def get_cached_hybrid_recommender(
    _content_model: ContentBasedRecommender,
    _collab_model: CollaborativeRecommender,
    _semantic_model: MovieSemanticSearch,
    movies_df: pd.DataFrame,
    ratings_df: Optional[pd.DataFrame] = None,
) -> HybridRecommender:
    """
    Creates and caches the HybridRecommender instance across Streamlit sessions.

    Unhashable recommender instances (_content_model, _collab_model, _semantic_model)
    are prefixed with an underscore so Streamlit excludes them from hashing,
    while movies_df and ratings_df remain hashed to guarantee cache safety
    if the underlying dataset changes.

    Args:
        _content_model: Fitted ContentBasedRecommender (unhashed).
        _collab_model: Fitted CollaborativeRecommender (unhashed).
        _semantic_model: Initialized MovieSemanticSearch (unhashed).
        movies_df: Validated DataFrame of movies (hashed for cache invalidation).
        ratings_df: Optional validated DataFrame of ratings (hashed for cache invalidation).

    Returns:
        HybridRecommender: Assembled multi-model recommendation engine.
    """
    return HybridRecommender(
        content_model=_content_model,
        collab_model=_collab_model,
        semantic_model=_semantic_model,
        movies_df=movies_df,
        ratings_df=ratings_df,
    )

"""
test_hybrid_recommender.py
==========================
Unit tests for CineMatch AI Hybrid Recommendation module.
Covers multi-signal score aggregation, dynamic weight re-normalization,
seed movie exclusion, already-rated item filtering, truthful explanation generation,
and graceful cold-start handling.
"""

from __future__ import annotations

import unittest
import pandas as pd

from data_loader import load_movies, load_ratings, load_tags
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender
from semantic_search import MovieSemanticSearch
from hybrid_recommender import HybridRecommender, get_cached_hybrid_recommender


class TestHybridRecommender(unittest.TestCase):
    """Test suite for Tri-Model HybridRecommender."""

    @classmethod
    def setUpClass(cls):
        """Loads dataset and initializes all three component models."""
        cls.movies_df = load_movies("data")
        cls.ratings_df = load_ratings("data")
        cls.tags_df = load_tags("data")

        cls.content_model = ContentBasedRecommender(cls.movies_df, cls.tags_df).fit()
        cls.collab_model = CollaborativeRecommender(cls.ratings_df, cls.movies_df, min_ratings=3).fit()
        cls.semantic_model = MovieSemanticSearch(cls.movies_df, cls.tags_df, force_fallback=True)

        cls.hybrid = HybridRecommender(
            content_model=cls.content_model,
            collab_model=cls.collab_model,
            semantic_model=cls.semantic_model,
            movies_df=cls.movies_df,
            ratings_df=cls.ratings_df,
        )

    def test_combine_all_three_signals(self):
        """Verify hybrid recommendations combine content, collaborative, and semantic signals."""
        recs = self.hybrid.recommend(
            movie_id=1,  # Toy Story
            user_id=1,
            query="funny animated family adventure",
            top_n=6,
        )

        self.assertFalse(recs.empty)
        self.assertEqual(len(recs), 6)

        required_cols = [
            "movieId",
            "title",
            "genres",
            "hybrid_score",
            "hybrid_percentage",
            "content_score",
            "collab_score",
            "semantic_score",
            "explanation",
            "active_signals",
        ]
        for col in required_cols:
            self.assertIn(col, recs.columns)

        # Check score bounds
        for _, row in recs.iterrows():
            self.assertGreaterEqual(row["hybrid_score"], 0.0)
            self.assertLessEqual(row["hybrid_score"], 1.0)
            self.assertTrue(row["explanation"].startswith("Recommended because"))

    def test_missing_semantic_query_handled_gracefully(self):
        """Verify recommendations succeed when query is None or empty."""
        recs_none = self.hybrid.recommend(movie_id=1, user_id=1, query=None, top_n=5)
        self.assertFalse(recs_none.empty)
        self.assertEqual(len(recs_none), 5)
        self.assertTrue((recs_none["semantic_score"] == 0.0).all())

        recs_empty = self.hybrid.recommend(movie_id=1, user_id=1, query="   ", top_n=5)
        self.assertFalse(recs_empty.empty)

    def test_missing_user_handled_gracefully(self):
        """Verify recommendations succeed without a user ID (using item collaborative)."""
        recs_no_user = self.hybrid.recommend(movie_id=1, user_id=None, query="space", top_n=5)
        self.assertFalse(recs_no_user.empty)
        self.assertEqual(len(recs_no_user), 5)
        # Should contain collaborative item signals
        self.assertTrue((recs_no_user["collab_score"] >= 0.0).all())

    def test_cold_start_unknown_user_fallback(self):
        """Verify unknown user ID does not crash the system (falls back to item collaborative)."""
        recs = self.hybrid.recommend(movie_id=1, user_id=9999999, query=None, top_n=5)
        self.assertFalse(recs.empty)
        self.assertEqual(len(recs), 5)

    def test_exclude_query_movie_and_no_duplicates(self):
        """Verify the selected seed movie is never in results and all IDs are unique."""
        seed_id = 1
        recs = self.hybrid.recommend(movie_id=seed_id, user_id=1, query="animation", top_n=10)

        # Seed movie excluded
        self.assertNotIn(seed_id, recs["movieId"].values)

        # Zero duplicates
        self.assertEqual(len(recs["movieId"]), len(set(recs["movieId"])))

    def test_exclude_already_rated_movies_for_user(self):
        """Verify movies already rated by user are strictly excluded from user recommendations."""
        user_id = 1
        user_rated_ids = set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"])

        recs = self.hybrid.recommend(movie_id=1, user_id=user_id, top_n=10)

        for mid in recs["movieId"]:
            self.assertNotIn(
                mid,
                user_rated_ids,
                f"Movie ID {mid} was already rated by user {user_id} and must not be in recommendations.",
            )

    def test_custom_weight_adjustment(self):
        """Verify custom weights change component weighting properly."""
        # Pure content weighting
        content_heavy = self.hybrid.recommend(
            movie_id=1,
            query="space adventure",
            weights={"content": 1.0, "collab": 0.0, "semantic": 0.0},
            top_n=3,
        )
        self.assertFalse(content_heavy.empty)
        # In pure content mode, hybrid_score equals content_score
        for _, row in content_heavy.iterrows():
            self.assertAlmostEqual(row["hybrid_score"], row["content_score"], places=3)

    def test_truthful_explanations(self):
        """Verify explanations honestly represent contributing signals without false claims."""
        recs = self.hybrid.recommend(
            movie_id=1,
            user_id=1,
            query="adventure",
            top_n=3,
        )

        for _, row in recs.iterrows():
            exp = row["explanation"]
            self.assertTrue(len(exp) > 10)
            # If Content is in active signals, text should mention genre/thematic DNA
            if "Content" in row["active_signals"]:
                self.assertIn("thematic/genre DNA", exp)

    def test_get_cached_hybrid_recommender(self):
        """Verify get_cached_hybrid_recommender constructs valid HybridRecommender with underscored parameters."""
        engine = get_cached_hybrid_recommender(
            _content_model=self.content_model,
            _collab_model=self.collab_model,
            _semantic_model=self.semantic_model,
            movies_df=self.movies_df,
            ratings_df=self.ratings_df,
        )
        self.assertIsInstance(engine, HybridRecommender)
        # Test positional argument invocation as well
        pos_engine = get_cached_hybrid_recommender(
            self.content_model,
            self.collab_model,
            self.semantic_model,
            self.movies_df,
            self.ratings_df,
        )
        self.assertIsInstance(pos_engine, HybridRecommender)
        recs = engine.recommend(movie_id=1, top_n=3)
        self.assertEqual(len(recs), 3)


if __name__ == "__main__":
    unittest.main()


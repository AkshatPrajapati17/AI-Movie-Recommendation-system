"""
test_collaborative.py
=====================
Unit tests for CineMatch AI Collaborative Filtering module.
Covers sparse CSR matrix construction, item-item cosine similarity,
user personalized recommendations (with exclusion of already-rated items),
unknown user/movie validation, cold-start handling, and profile extraction.
"""

from __future__ import annotations

import unittest
import pandas as pd

from data_loader import load_movies, load_ratings
from collaborative import CollaborativeRecommender


class TestCollaborativeRecommender(unittest.TestCase):
    """Test suite for CollaborativeRecommender."""

    @classmethod
    def setUpClass(cls):
        """Loads dataset and fits the collaborative recommender once for all tests."""
        cls.movies_df = load_movies("data")
        cls.ratings_df = load_ratings("data")
        cls.recommender = CollaborativeRecommender(cls.ratings_df, cls.movies_df, min_ratings=3)
        cls.recommender.fit()

    def test_fit_and_matrix_shapes(self):
        """Verify sparse rating matrices are correctly constructed."""
        self.assertTrue(self.recommender._is_fitted)
        self.assertIsNotNone(self.recommender.item_user_matrix)
        self.assertIsNotNone(self.recommender.norm_item_matrix)
        self.assertEqual(self.recommender.item_user_matrix.shape[0], len(self.recommender.movie_ids))
        self.assertEqual(self.recommender.item_user_matrix.shape[1], len(self.recommender.user_ids))
        self.assertEqual(self.recommender.item_user_matrix.nnz, len(self.ratings_df))

    def test_similar_movies_excludes_query_movie(self):
        """Verify item similarity for Toy Story (movieId=1) excludes Toy Story itself."""
        toy_story_id = 1
        sim_df = self.recommender.get_similar_movies(toy_story_id, top_n=5)

        self.assertFalse(sim_df.empty)
        self.assertEqual(len(sim_df), 5)
        self.assertNotIn(toy_story_id, sim_df["movieId"].values)
        self.assertNotIn("Toy Story (1995)", sim_df["title"].values)

        # Expected top similar items include Toy Story 2 or Jurassic Park
        top_titles = " ".join(sim_df["title"].tolist())
        self.assertTrue(
            "Toy Story 2" in top_titles or "Jurassic Park" in top_titles or "Independence Day" in top_titles,
            f"Expected familiar MovieLens co-rated blockbusters, got: {top_titles}",
        )

    def test_recommend_for_user_excludes_rated_movies(self):
        """Verify recommendations for User 1 strictly exclude movies User 1 already rated."""
        user_id = 1
        user_rated_ids = set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"])

        user_recs = self.recommender.recommend_for_user(user_id=user_id, top_n=10)

        self.assertFalse(user_recs.empty)
        self.assertEqual(len(user_recs), 10)

        # Assert zero overlap with already-rated items
        for mid in user_recs["movieId"]:
            self.assertNotIn(
                mid,
                user_rated_ids,
                f"Movie ID {mid} was already rated by user {user_id} and must not be recommended.",
            )

    def test_recommend_for_user_required_columns(self):
        """Verify user recommendation output contains required fields."""
        recs = self.recommender.recommend_for_user(user_id=1, top_n=5)
        for col in ["movieId", "title", "genres", "collaborative_score", "affinity_percentage"]:
            self.assertIn(col, recs.columns)

    def test_unknown_user_raises_value_error(self):
        """Verify ValueError is raised when querying a non-existent user ID."""
        with self.assertRaises(ValueError) as ctx:
            self.recommender.recommend_for_user(user_id=9999999)
        self.assertIn("not found in the ratings dataset", str(ctx.exception))

    def test_unknown_movie_raises_value_error(self):
        """Verify ValueError is raised when querying a non-existent movie ID."""
        with self.assertRaises(ValueError) as ctx:
            self.recommender.get_similar_movies(movie_id=9999999)
        self.assertIn("not found in the movie catalog", str(ctx.exception))

    def test_unrated_movie_handled_gracefully(self):
        """Verify movies with zero ratings in ratings.csv return empty DataFrame without error."""
        rated_ids = set(self.ratings_df["movieId"])
        all_ids = set(self.movies_df["movieId"])
        unrated_ids = list(all_ids - rated_ids)

        if unrated_ids:
            unrated_mid = unrated_ids[0]
            res = self.recommender.get_similar_movies(unrated_mid)
            self.assertTrue(res.empty)

    def test_user_profile_extraction(self):
        """Verify user profile extraction retrieves accurate rating metrics and favorites."""
        profile = self.recommender.get_user_profile(user_id=1)

        self.assertEqual(profile["user_id"], 1)
        self.assertEqual(profile["total_ratings"], 232)
        self.assertAlmostEqual(profile["avg_rating"], 4.37, delta=0.1)
        self.assertEqual(len(profile["top_movies"]), 5)
        self.assertTrue(len(profile["top_genres"]) > 0)


if __name__ == "__main__":
    unittest.main()

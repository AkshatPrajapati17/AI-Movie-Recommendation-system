"""
test_content_based.py
=====================
Unit tests for CineMatch AI Content-Based Filtering module.
Covers TF-IDF vectorization, cosine similarity ranking, exclusions of query movie,
unknown movie ID handling, empty metadata edge cases, and score bounds.
"""

from __future__ import annotations

import unittest
import pandas as pd

from data_loader import load_movies, load_tags
from content_based import ContentBasedRecommender


class TestContentBasedRecommender(unittest.TestCase):
    """Test suite for ContentBasedRecommender."""

    @classmethod
    def setUpClass(cls):
        """Loads and fits the model once for testing against the real MovieLens dataset."""
        cls.movies_df = load_movies("data")
        cls.tags_df = load_tags("data")
        cls.recommender = ContentBasedRecommender(cls.movies_df, cls.tags_df)
        cls.recommender.fit()

    def test_fit_and_matrix_dimensions(self):
        """Verify TF-IDF matrix is created with correct dimensions."""
        self.assertTrue(self.recommender._is_fitted)
        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertEqual(self.recommender.tfidf_matrix.shape[0], len(self.movies_df))
        self.assertGreater(self.recommender.tfidf_matrix.shape[1], 100)

    def test_recommend_excludes_query_movie(self):
        """Verify recommendations for Toy Story (movieId=1) do NOT include Toy Story."""
        toy_story_id = 1
        recs = self.recommender.recommend(movie_id=toy_story_id, top_n=5)

        self.assertFalse(recs.empty)
        self.assertEqual(len(recs), 5)
        self.assertNotIn(toy_story_id, recs["movieId"].values)
        self.assertNotIn("Toy Story (1995)", recs["title"].values)

    def test_recommend_required_columns(self):
        """Verify returned DataFrame has all required presentation fields."""
        recs = self.recommender.recommend(movie_id=1, top_n=5)
        for col in ["movieId", "title", "genres", "similarity_score", "similarity_percentage"]:
            self.assertIn(col, recs.columns)

    def test_similarity_score_bounds(self):
        """Verify similarity scores are valid floats within [0.0, 1.0]."""
        recs = self.recommender.recommend(movie_id=1, top_n=10)
        for _, row in recs.iterrows():
            score = row["similarity_score"]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            self.assertTrue(row["similarity_percentage"].endswith("%"))

    def test_recommend_descending_order(self):
        """Verify recommendations are sorted from highest to lowest similarity."""
        recs = self.recommender.recommend(movie_id=1, top_n=10)
        scores = recs["similarity_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unknown_movie_id_raises_value_error(self):
        """Verify ValueError is raised for non-existent movie IDs."""
        with self.assertRaises(ValueError) as ctx:
            self.recommender.recommend(movie_id=99999999)
        self.assertIn("not found in the movie catalog", str(ctx.exception))

    def test_unfitted_model_raises_value_error(self):
        """Verify calling recommend() before fit() raises ValueError."""
        unfitted = ContentBasedRecommender(self.movies_df, self.tags_df)
        with self.assertRaises(ValueError) as ctx:
            unfitted.recommend(movie_id=1)
        self.assertIn("must be fitted", str(ctx.exception))

    def test_movie_with_no_usable_features(self):
        """Verify that a movie with no genres or tags is handled gracefully without crashing."""
        minimal_movies = pd.DataFrame({
            "movieId": [101, 102],
            "title": ["Empty Movie (2020)", "Action Movie (2020)"],
            "genres": ["(no genres listed)", "Action|Thriller"],
            "year": [2020, 2020],
        })
        model = ContentBasedRecommender(minimal_movies, tags_df=None).fit()

        # Movie 101 has zero usable features
        self.assertFalse(model.has_usable_features(101))
        self.assertTrue(model.has_usable_features(102))

        # Recommending on movie 101 should return empty DataFrame gracefully
        recs_empty = model.recommend(101)
        self.assertTrue(recs_empty.empty)


if __name__ == "__main__":
    unittest.main()

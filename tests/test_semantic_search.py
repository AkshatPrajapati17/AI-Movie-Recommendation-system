"""
test_semantic_search.py
=======================
Unit tests for CineMatch AI Semantic Movie Search module.
Covers ranking behavior, empty queries, missing fields, cache invalidation,
natural language matching, and graceful TF-IDF fallback.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
import pandas as pd

from data_loader import load_movies, load_tags
from semantic_search import MovieSemanticSearch


class TestSemanticSearch(unittest.TestCase):
    """Test suite for MovieSemanticSearch engine and fallback modes."""

    @classmethod
    def setUpClass(cls):
        """Loads dataset and initializes search engine."""
        cls.movies_df = load_movies("data")
        cls.tags_df = load_tags("data")
        # Use fallback mode for fast, deterministic unit test execution
        cls.fallback_searcher = MovieSemanticSearch(
            movies_df=cls.movies_df,
            tags_df=cls.tags_df,
            force_fallback=True,
        )

    def test_empty_query_returns_empty_dataframe(self):
        """Verify empty and whitespace queries return empty DataFrame with proper schema."""
        res_empty = self.fallback_searcher.search("")
        self.assertTrue(res_empty.empty)
        for col in ["movieId", "title", "genres", "similarity_score", "similarity_percentage"]:
            self.assertIn(col, res_empty.columns)

        res_spaces = self.fallback_searcher.search("     ")
        self.assertTrue(res_spaces.empty)

    def test_ranking_descending_order(self):
        """Verify search results are sorted in descending order of similarity."""
        query = "space adventure"
        results = self.fallback_searcher.search(query, top_n=8)

        self.assertFalse(results.empty)
        scores = results["similarity_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
        for _, row in results.iterrows():
            self.assertTrue(row["similarity_percentage"].endswith("%"))

    def test_natural_language_queries(self):
        """Verify natural language thematic queries return relevant genre films."""
        # Query 1: Science fiction space
        sf_results = self.fallback_searcher.search("science fiction space adventure", top_n=5)
        self.assertFalse(sf_results.empty)
        sf_genres = " ".join(sf_results["genres"].tolist())
        self.assertTrue(
            "Sci-Fi" in sf_genres or "Adventure" in sf_genres,
            f"Expected Sci-Fi or Adventure in {sf_genres}",
        )

        # Query 2: Crime thriller
        crime_results = self.fallback_searcher.search("dark crime thrillers", top_n=5)
        self.assertFalse(crime_results.empty)
        crime_genres = " ".join(crime_results["genres"].tolist())
        self.assertTrue(
            "Crime" in crime_genres or "Thriller" in crime_genres,
            f"Expected Crime or Thriller in {crime_genres}",
        )

    def test_fallback_activation_when_forced(self):
        """Verify engine flags and status when TF-IDF fallback is active."""
        searcher = MovieSemanticSearch(
            self.movies_df,
            self.tags_df,
            force_fallback=True,
        )
        self.assertTrue(searcher.is_fallback)
        self.assertIn("TF-IDF", searcher.engine_type)
        self.assertIsNotNone(searcher.fallback_reason)

        # Confirm queries still return valid results in fallback mode
        res = searcher.search("Toy Story", top_n=3)
        self.assertFalse(res.empty)
        self.assertIn("Toy Story", res.iloc[0]["title"])

    def test_missing_fields_handled_gracefully(self):
        """Verify movies with no genres or missing tags do not crash the engine."""
        sparse_movies = pd.DataFrame({
            "movieId": [901, 902],
            "title": ["Ghost Movie (2020)", "Mystery Movie (2021)"],
            "genres": ["(no genres listed)", "Mystery"],
            "year": [None, 2021],
        })
        searcher = MovieSemanticSearch(sparse_movies, tags_df=None, force_fallback=True)

        res = searcher.search("Mystery", top_n=2)
        self.assertFalse(res.empty)
        self.assertEqual(res.iloc[0]["movieId"], 902)

    def test_cache_invalidation_on_data_change(self):
        """Verify cache invalidation logic detects modified corpus or movie counts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            searcher1 = MovieSemanticSearch(
                self.movies_df.head(20),
                tags_df=None,
                cache_dir=tmp_dir,
                force_fallback=True,
            )
            hash1 = searcher1._compute_corpus_hash()

            # Second searcher with different data length
            searcher2 = MovieSemanticSearch(
                self.movies_df.head(30),
                tags_df=None,
                cache_dir=tmp_dir,
                force_fallback=True,
            )
            hash2 = searcher2._compute_corpus_hash()

            self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()

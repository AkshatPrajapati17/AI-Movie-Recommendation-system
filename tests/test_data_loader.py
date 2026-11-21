"""
test_data_loader.py
===================
Unit tests for CineMatch AI data loader and validation module.
Covers schema validation, duplicate detection, missing optional files,
error conditions, and statistical profiling.

Runs seamlessly under both `pytest tests` and `python -m unittest discover tests`.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import pandas as pd

try:
    import pytest
except ImportError:
    import contextlib

    class _ExceptionInfo:
        def __init__(self, val=None):
            self.value = val

    class _PytestFallback:
        @staticmethod
        def raises(expected_exception):
            @contextlib.contextmanager
            def _cm():
                info = _ExceptionInfo()
                try:
                    yield info
                except expected_exception as exc:
                    info.value = exc
                else:
                    raise AssertionError(f"Expected exception {expected_exception} was not raised.")
            return _cm()

    pytest = _PytestFallback()

from data_loader import (
    DataValidationError,
    DatasetNotFoundError,
    compute_dataset_statistics,
    enrich_movie_catalog,
    load_and_validate_dataset,
    load_links,
    load_movies,
    load_ratings,
    load_tags,
    validate_movies_dataframe,
    validate_ratings_dataframe,
    validate_tags_dataframe,
)


class TestDataLoader(unittest.TestCase):
    """Automated test cases for data_loader functions and integrity validations."""

    def test_load_real_dataset(self):
        """Verify loading and profiling on the local MovieLens dataset."""
        movies_df, ratings_df, tags_df, links_df = load_and_validate_dataset("data")

        self.assertFalse(movies_df.empty, "movies_df should not be empty")
        self.assertEqual(len(movies_df), 9742)
        self.assertIn("movieId", movies_df.columns)
        self.assertIn("title", movies_df.columns)
        self.assertIn("genres", movies_df.columns)
        self.assertIn("year", movies_df.columns)

        self.assertFalse(ratings_df.empty, "ratings_df should not be empty")
        self.assertEqual(len(ratings_df), 100836)
        self.assertGreaterEqual(ratings_df["rating"].min(), 0.5)
        self.assertLessEqual(ratings_df["rating"].max(), 5.0)

        self.assertIsNotNone(tags_df)
        self.assertEqual(len(tags_df), 3683)

        self.assertIsNotNone(links_df)
        self.assertEqual(len(links_df), 9742)

    def test_statistics_calculation(self):
        """Verify statistical profiling produces accurate, non-fabricated metrics."""
        movies_df = load_movies("data")
        ratings_df = load_ratings("data")
        tags_df = load_tags("data")

        stats = compute_dataset_statistics(movies_df, ratings_df, tags_df)

        self.assertEqual(stats["total_movies"], 9742)
        self.assertEqual(stats["total_ratings"], 100836)
        self.assertEqual(stats["unique_users"], 610)
        self.assertEqual(stats["average_rating"], 3.5)
        self.assertEqual(stats["unique_genres_count"], 19)
        self.assertIn("Drama", stats["genre_counts"])
        self.assertIn("Comedy", stats["genre_counts"])
        self.assertTrue(stats["has_tags"])

    def test_enrich_movie_catalog(self):
        """Verify catalog enrichment accurately aggregates real ratings without fabrication."""
        movies_df = load_movies("data")
        ratings_df = load_ratings("data")
        tags_df = load_tags("data")

        enriched = enrich_movie_catalog(movies_df, ratings_df, tags_df)

        self.assertIn("rating_count", enriched.columns)
        self.assertIn("avg_rating", enriched.columns)
        self.assertIn("sample_tags", enriched.columns)
        self.assertEqual(len(enriched), len(movies_df))

        # Check Forrest Gump (movieId=356 in MovieLens)
        gump = enriched[enriched["movieId"] == 356]
        self.assertFalse(gump.empty)
        self.assertEqual(gump["rating_count"].iloc[0], 329)
        self.assertEqual(gump["avg_rating"].iloc[0], 4.16)

    def test_missing_dataset_directory(self):
        """Verify friendly DatasetNotFoundError when data directory does not exist."""
        with self.assertRaises(DatasetNotFoundError) as ctx:
            load_movies("non_existent_folder_xyz")
        self.assertIn("movies.csv", str(ctx.exception))

    def test_missing_required_column_in_movies(self):
        """Verify DataValidationError when a required column is missing."""
        malformed_df = pd.DataFrame({
            "movieId": [1, 2],
            "genres": ["Action", "Comedy"],
            # Missing 'title'
        })
        with self.assertRaises(DataValidationError) as ctx:
            validate_movies_dataframe(malformed_df)
        self.assertIn("title", str(ctx.exception))

    def test_null_movie_id_validation(self):
        """Verify DataValidationError when movieId contains null values."""
        malformed_df = pd.DataFrame({
            "movieId": [1, None, 3],
            "title": ["Movie A", "Movie B", "Movie C"],
            "genres": ["Action", "Drama", "Comedy"],
        })
        with self.assertRaises(DataValidationError) as ctx:
            validate_movies_dataframe(malformed_df)
        self.assertIn("null", str(ctx.exception).lower())

    def test_duplicate_movie_id_validation(self):
        """Verify DataValidationError when duplicate movieIds are present."""
        malformed_df = pd.DataFrame({
            "movieId": [1, 2, 2],
            "title": ["Movie A", "Movie B", "Movie B (Remake)"],
            "genres": ["Action", "Drama", "Comedy"],
        })
        with self.assertRaises(DataValidationError) as ctx:
            validate_movies_dataframe(malformed_df)
        self.assertIn("Duplicate movie IDs", str(ctx.exception))

    def test_malformed_ratings_columns(self):
        """Verify DataValidationError when ratings dataset lacks required columns."""
        malformed_ratings = pd.DataFrame({
            "userId": [1, 2],
            "rating": [4.0, 5.0],
            # Missing 'movieId' and 'timestamp'
        })
        with self.assertRaises(DataValidationError) as ctx:
            validate_ratings_dataframe(malformed_ratings)
        self.assertIn("Missing required column", str(ctx.exception))

    def test_out_of_bounds_ratings(self):
        """Verify DataValidationError when rating values exceed permissible scale."""
        invalid_ratings = pd.DataFrame({
            "userId": [1],
            "movieId": [10],
            "rating": [99.0],  # Out of bounds
            "timestamp": [123456789],
        })
        with self.assertRaises(DataValidationError) as ctx:
            validate_ratings_dataframe(invalid_ratings)
        self.assertIn("bounds", str(ctx.exception).lower())

    def test_optional_tags_missing_gracefully(self):
        """Verify that a missing tags.csv does not crash the loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal valid movies.csv and ratings.csv
            movies_df = pd.DataFrame({
                "movieId": [1, 2],
                "title": ["Toy Story (1995)", "Jumanji (1995)"],
                "genres": ["Animation|Comedy", "Adventure|Children"],
            })
            ratings_df = pd.DataFrame({
                "userId": [1, 1],
                "movieId": [1, 2],
                "rating": [5.0, 4.0],
                "timestamp": [964982703, 964982704],
            })
            movies_df.to_csv(os.path.join(temp_dir, "movies.csv"), index=False)
            ratings_df.to_csv(os.path.join(temp_dir, "ratings.csv"), index=False)

            # tags.csv is purposely omitted
            loaded_tags = load_tags(temp_dir)
            self.assertIsNone(loaded_tags, "Missing tags.csv should gracefully return None")

            # load_and_validate_dataset should succeed with tags_df = None
            m, r, t, l = load_and_validate_dataset(temp_dir)
            self.assertEqual(len(m), 2)
            self.assertEqual(len(r), 2)
            self.assertIsNone(t)
            self.assertIsNone(l)


if __name__ == "__main__":
    unittest.main()

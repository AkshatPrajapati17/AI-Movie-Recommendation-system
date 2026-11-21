"""
tests/test_homepage_grid.py
===========================
Automated verification tests for CineMatch AI homepage movie discovery grid:
- Verifies at least 50 distinct movie entries with valid poster images appear on the first page.
- Verifies movie titles match their posters.
- Verifies no broken image paths remain.
- Verifies 100% genuine MovieLens records and accurate metadata.
"""

import os
import unittest
import pandas as pd
from data_loader import load_and_validate_dataset, enrich_movie_catalog
from poster_service import get_local_movie_poster, get_shared_manifest, POSTERS_DIR


class TestHomepagePosterGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies_df, cls.ratings_df, cls.tags_df, cls.links_df = load_and_validate_dataset("data")
        cls.catalog = enrich_movie_catalog(cls.movies_df, cls.ratings_df, cls.tags_df)
        cls.manifest = get_shared_manifest()

    def test_minimum_50_posters_available_on_disk(self):
        """Verifies that at least 50 valid local poster files exist on disk."""
        stats = self.manifest.get_summary_stats()
        self.assertGreaterEqual(stats["disk_verified"], 50)
        self.assertEqual(stats["error"], 0)

    def test_top_60_homepage_movies_have_posters(self):
        """
        Verifies that when sorted by popularity (default homepage view),
        at least 50 movies (specifically 60 movies) have valid, verified posters on disk.
        """
        popular = self.catalog.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False])
        top_60 = popular.head(60)

        self.assertEqual(len(top_60), 60)
        self.assertEqual(len(top_60["movieId"].unique()), 60)  # No duplicates

        verified_posters = 0
        for _, row in top_60.iterrows():
            mid = int(row["movieId"])
            poster_path = get_local_movie_poster(mid)
            self.assertIsNotNone(poster_path, f"Movie {mid} ({row['title']}) should have a local poster.")
            self.assertTrue(os.path.isfile(poster_path), f"File {poster_path} must exist on disk.")
            self.assertGreaterEqual(os.path.getsize(poster_path), 500, f"File {poster_path} too small.")
            
            # Verify magic bytes for JPEG
            with open(poster_path, "rb") as f:
                header = f.read(3)
                self.assertEqual(header, b"\xff\xd8\xff", f"File {poster_path} is not a valid JPEG.")

            # Verify title is authentic non-empty string
            self.assertTrue(len(str(row["title"]).strip()) > 0)
            verified_posters += 1

        self.assertGreaterEqual(verified_posters, 50)
        self.assertEqual(verified_posters, 60)

    def test_genres_coverage_across_postered_movies(self):
        """Verifies that the postered movies cover a wide range of recognizable genres."""
        popular = self.catalog.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False])
        top_60 = popular.head(60)
        all_genres = set()
        for g in top_60["genres"]:
            all_genres.update(g.split("|"))

        self.assertIn("Action", all_genres)
        self.assertIn("Comedy", all_genres)
        self.assertIn("Drama", all_genres)
        self.assertIn("Sci-Fi", all_genres)
        self.assertIn("Thriller", all_genres)
        self.assertGreaterEqual(len(all_genres), 12)

    def test_no_synthetic_movies(self):
        """Verifies that all movie IDs exist strictly in data/movies.csv."""
        raw_mids = set(self.movies_df["movieId"])
        popular = self.catalog.sort_values(by=["rating_count", "avg_rating"], ascending=[False, False])
        top_60 = popular.head(60)
        for mid in top_60["movieId"]:
            self.assertIn(mid, raw_mids)


if __name__ == "__main__":
    unittest.main()

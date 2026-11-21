"""
test_poster_service.py
======================
Unit tests for CineMatch AI poster acquisition, title normalization,
manifest management, and TMDb client verification.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from poster_service import (
    PosterManifest,
    TMDbPosterService,
    get_local_movie_poster,
    get_tmdb_api_key,
    is_api_key_configured,
    parse_movielens_title,
)


class TestTitleParsing(unittest.TestCase):
    """Verifies that MovieLens titles are normalized accurately without hallucinating."""

    def test_standard_title_and_year(self):
        clean, year, alts = parse_movielens_title("Toy Story (1995)")
        self.assertEqual(clean, "Toy Story")
        self.assertEqual(year, 1995)
        self.assertEqual(alts, [])

    def test_inverted_article_the(self):
        clean, year, alts = parse_movielens_title("American President, The (1995)")
        self.assertEqual(clean, "The American President")
        self.assertEqual(year, 1995)
        self.assertEqual(alts, [])

    def test_inverted_article_french_la(self):
        clean, year, alts = parse_movielens_title("City of Lost Children, The (Cité des enfants perdus, La) (1995)")
        self.assertEqual(clean, "The City of Lost Children")
        self.assertEqual(year, 1995)
        self.assertIn("La Cité des enfants perdus", alts)

    def test_aka_title_extraction(self):
        clean, year, alts = parse_movielens_title("Seven (a.k.a. Se7en) (1995)")
        self.assertEqual(clean, "Seven")
        self.assertEqual(year, 1995)
        self.assertIn("Se7en", alts)

    def test_italian_article_il(self):
        clean, year, alts = parse_movielens_title("Postman, The (Postino, Il) (1994)")
        self.assertEqual(clean, "The Postman")
        self.assertEqual(year, 1994)
        self.assertIn("Il Postino", alts)

    def test_missing_year_graceful_handling(self):
        clean, year, alts = parse_movielens_title("Babylon 5")
        self.assertEqual(clean, "Babylon 5")
        self.assertIsNone(year)
        self.assertEqual(alts, [])


class TestPosterManifest(unittest.TestCase):
    """Verifies manifest persistence, lookup, and atomic saving."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manifest_file = os.path.join(self.test_dir, "manifest.json")
        self.manifest = PosterManifest(manifest_path=self.manifest_file, posters_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_manifest_initialization(self):
        self.assertEqual(len(self.manifest.entries), 0)
        self.assertIsNone(self.manifest.get(1))
        self.assertIsNone(self.manifest.get_local_poster_path(1))

    def test_record_downloaded_and_lookup(self):
        poster_file = os.path.join(self.test_dir, "1.jpg")
        # Write dummy JPEG content (> 100 bytes)
        with open(poster_file, "wb") as f:
            f.write(b"\xff\xd8\xff" + b"0" * 200)

        self.manifest.record_downloaded(
            movie_id=1,
            title="Toy Story (1995)",
            clean_title="Toy Story",
            year=1995,
            tmdb_id=862,
            imdb_id="0114709",
            poster_path="/rhIRbist9RLGMs1CU2i82HLAkgQ.jpg",
            local_path=poster_file,
            match_method="tmdb_id",
        )
        self.manifest.save()

        # Reload manifest in fresh instance
        fresh_manifest = PosterManifest(manifest_path=self.manifest_file, posters_dir=self.test_dir)
        entry = fresh_manifest.get(1)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["status"], "downloaded")
        self.assertEqual(entry["tmdb_id"], 862)
        self.assertEqual(fresh_manifest.get_local_poster_path(1), poster_file)

    def test_record_no_poster_and_not_found(self):
        self.manifest.record_no_poster(
            movie_id=2,
            title="Obscure Indie (2000)",
            clean_title="Obscure Indie",
            year=2000,
            tmdb_id=99999,
            imdb_id=None,
            match_method="title_year_search",
        )
        self.manifest.record_not_found(
            movie_id=3,
            title="Unmatched Movie (1990)",
            clean_title="Unmatched Movie",
            year=1990,
            tmdb_id=None,
            imdb_id=None,
        )
        self.manifest.save()

        stats = self.manifest.get_summary_stats()
        self.assertEqual(stats["total_recorded"], 2)
        self.assertEqual(stats["no_poster"], 1)
        self.assertEqual(stats["not_found"], 1)
        self.assertIsNone(self.manifest.get_local_poster_path(2))
        self.assertIsNone(self.manifest.get_local_poster_path(3))

    def test_corrupt_poster_file_rejected(self):
        corrupt_file = os.path.join(self.test_dir, "50.jpg")
        with open(corrupt_file, "w") as f:
            f.write("<html>404 Not Found</html>" * 10)
        self.manifest.record_downloaded(
            movie_id=50,
            title="Corrupt Film (1995)",
            clean_title="Corrupt Film",
            year=1995,
            tmdb_id=123,
            imdb_id="0123456",
            poster_path="/corrupt.jpg",
            local_path=corrupt_file,
            match_method="tmdb_id",
        )
        self.assertIsNone(self.manifest.get_local_poster_path(50))

    def test_manifest_check_reload(self):
        m2 = PosterManifest(manifest_path=self.manifest_file, posters_dir=self.test_dir)
        self.assertEqual(len(m2.entries), 0)

        self.manifest.record_not_found(
            movie_id=77,
            title="Test (2000)",
            clean_title="Test",
            year=2000,
            tmdb_id=None,
            imdb_id=None,
        )
        self.manifest.save()

        m2.check_reload()
        self.assertIsNotNone(m2.get(77))



class TestAPIKeyHelpers(unittest.TestCase):
    """Verifies API key configuration detection and security."""

    def test_placeholder_rejection(self):
        self.assertIsNone(get_tmdb_api_key("your_tmdb_api_key_here"))
        self.assertIsNone(get_tmdb_api_key(""))
        self.assertIsNone(get_tmdb_api_key("none"))
        self.assertFalse(is_api_key_configured("your_tmdb_api_key_here"))

    def test_valid_key_acceptance(self):
        self.assertEqual(get_tmdb_api_key("abc123xyz789"), "abc123xyz789")
        self.assertTrue(is_api_key_configured("abc123xyz789"))


class TestTMDbPosterServiceMatching(unittest.TestCase):
    """Verifies tiered matching logic using mocked HTTP responses."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manifest = PosterManifest(
            manifest_path=os.path.join(self.test_dir, "manifest.json"),
            posters_dir=self.test_dir,
        )
        self.service = TMDbPosterService(
            api_key="mock_valid_key_12345",
            posters_dir=self.test_dir,
            manifest=self.manifest,
            request_delay=0.0,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(TMDbPosterService, "download_poster_image", return_value=True)
    @patch.object(TMDbPosterService, "query_by_tmdb_id")
    def test_tier1_tmdb_id_match(self, mock_query, mock_dl):
        mock_query.return_value = {
            "id": 862,
            "title": "Toy Story",
            "poster_path": "/rhIRbist9RLGMs1CU2i82HLAkgQ.jpg",
        }

        result = self.service.match_and_fetch_poster(
            movie_id=1,
            title="Toy Story (1995)",
            year=1995,
            tmdb_id=862,
            imdb_id="0114709",
        )

        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(result["match_method"], "tmdb_id")
        self.assertEqual(result["poster_path"], "/rhIRbist9RLGMs1CU2i82HLAkgQ.jpg")
        mock_dl.assert_called_once()

    @patch.object(TMDbPosterService, "download_poster_image", return_value=True)
    @patch.object(TMDbPosterService, "query_by_tmdb_id", return_value=None)
    @patch.object(TMDbPosterService, "query_by_imdb_id")
    def test_tier2_imdb_id_fallback(self, mock_imdb, mock_tmdb, mock_dl):
        mock_imdb.return_value = {
            "id": 862,
            "title": "Toy Story",
            "poster_path": "/rhIRbist9RLGMs1CU2i82HLAkgQ.jpg",
        }

        result = self.service.match_and_fetch_poster(
            movie_id=1,
            title="Toy Story (1995)",
            year=1995,
            tmdb_id=None,
            imdb_id="0114709",
        )

        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(result["match_method"], "imdb_id")
        mock_dl.assert_called_once()

    @patch.object(TMDbPosterService, "download_poster_image", return_value=True)
    @patch.object(TMDbPosterService, "query_by_tmdb_id", return_value=None)
    @patch.object(TMDbPosterService, "query_by_imdb_id", return_value=None)
    @patch.object(TMDbPosterService, "search_by_title_and_year")
    def test_tier3_search_title_year_fallback(self, mock_search, mock_imdb, mock_tmdb, mock_dl):
        mock_search.return_value = [{
            "id": 862,
            "title": "Toy Story",
            "release_date": "1995-10-30",
            "poster_path": "/rhIRbist9RLGMs1CU2i82HLAkgQ.jpg",
        }]

        result = self.service.match_and_fetch_poster(
            movie_id=1,
            title="Toy Story (1995)",
            year=1995,
            tmdb_id=None,
            imdb_id=None,
        )

        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(result["match_method"], "title_year_search")
        mock_dl.assert_called_once()

    @patch.object(TMDbPosterService, "query_by_tmdb_id")
    def test_matched_movie_without_poster(self, mock_query):
        mock_query.return_value = {
            "id": 99999,
            "title": "Posterless Movie",
            "poster_path": None,
        }

        result = self.service.match_and_fetch_poster(
            movie_id=99,
            title="Posterless Movie (2005)",
            year=2005,
            tmdb_id=99999,
        )

        self.assertEqual(result["status"], "no_poster")
        self.assertIsNone(result["poster_path"])
        self.assertIsNone(result["local_path"])

    @patch.object(TMDbPosterService, "query_by_tmdb_id", return_value=None)
    @patch.object(TMDbPosterService, "query_by_imdb_id", return_value=None)
    @patch.object(TMDbPosterService, "search_by_title_and_year", return_value=[])
    def test_unmatched_movie_not_found(self, mock_s, mock_i, mock_t):
        result = self.service.match_and_fetch_poster(
            movie_id=100,
            title="Completely Nonexistent Movie XYZ (1920)",
            year=1920,
        )

        self.assertEqual(result["status"], "not_found")
        self.assertIsNone(result["poster_path"])

    @patch("poster_service.PosterManifest")
    @patch.object(TMDbPosterService, "match_and_fetch_poster")
    def test_batch_fetch_posters(self, mock_match, mock_manifest_cls):
        from poster_service import batch_fetch_posters

        mock_manifest_inst = MagicMock()
        mock_manifest_inst.get.return_value = None
        mock_manifest_cls.return_value = mock_manifest_inst

        mock_match.return_value = {
            "movie_id": 1,
            "status": "downloaded",
            "local_path": "posters/1.jpg",
        }

        test_movies = pd.DataFrame([
            {"movieId": 1, "title": "Toy Story (1995)", "year": 1995},
            {"movieId": 2, "title": "Jumanji (1995)", "year": 1995},
        ])

        summary = batch_fetch_posters(
            movies_df=test_movies,
            api_key="mock_valid_key",
            limit=2,
            delay=0.0,
        )

        self.assertEqual(summary["total_queued"], 2)
        self.assertEqual(summary["processed"], 2)
        self.assertEqual(summary["downloaded"], 2)
        self.assertEqual(mock_match.call_count, 2)


class TestOfflineHelper(unittest.TestCase):
    """Verifies get_local_movie_poster never errors and performs fast offline lookup."""

    def test_get_local_movie_poster_nonexistent(self):
        # Movie 999999 does not exist on disk
        res = get_local_movie_poster(999999)
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

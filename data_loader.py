
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Optional Streamlit caching decorator helper
try:
    import streamlit as st
    cache_decorator = st.cache_data
except Exception:  # pragma: no cover
    def cache_decorator(*args, **kwargs):
        def inner(func):
            return func
        return inner


# ---------------------------------------------------------------------------
# Custom Exceptions for Clear, Friendly Error Handling
# ---------------------------------------------------------------------------

class DataLoadError(Exception):
    """Base exception for CineMatch data loading errors."""
    pass


class DatasetNotFoundError(DataLoadError):
    """Raised when an essential dataset file (e.g., movies.csv or ratings.csv) cannot be found."""
    pass


class DataValidationError(DataLoadError):
    """Raised when dataset fails schema, type, or integrity validation."""
    pass


# ---------------------------------------------------------------------------
# Schema and Validation Definitions
# ---------------------------------------------------------------------------

REQUIRED_MOVIE_COLUMNS: Set[str] = {"movieId", "title", "genres"}
REQUIRED_RATING_COLUMNS: Set[str] = {"userId", "movieId", "rating", "timestamp"}
REQUIRED_TAG_COLUMNS: Set[str] = {"userId", "movieId", "tag"}
REQUIRED_LINK_COLUMNS: Set[str] = {"movieId", "imdbId"}


def validate_movies_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the structure and data types of the movies dataset.

    Checks:
    1. Presence of required columns ('movieId', 'title', 'genres').
    2. Absence of null values in critical identity columns ('movieId', 'title').
    3. Absence of duplicate 'movieId' keys.
    4. Proper casting to expected data types.
    5. Clean extraction of release years without fabricating titles.

    Args:
        df: Raw DataFrame read from movies.csv.

    Returns:
        pd.DataFrame: Cleaned and validated movies DataFrame.

    Raises:
        DataValidationError: If any critical validation fails.
    """
    if df.empty:
        raise DataValidationError("The movies dataset is empty.")

    # 1. Required columns check
    missing_cols = REQUIRED_MOVIE_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Malformed movies dataset. Missing required column(s): {sorted(list(missing_cols))}. "
            f"Expected columns: {sorted(list(REQUIRED_MOVIE_COLUMNS))}."
        )

    # 2. Null checks
    null_id_count = int(df["movieId"].isnull().sum())
    null_title_count = int(df["title"].isnull().sum())
    if null_id_count > 0:
        raise DataValidationError(f"Found {null_id_count} null or missing value(s) in 'movieId'.")
    if null_title_count > 0:
        raise DataValidationError(f"Found {null_title_count} null or missing value(s) in 'title'.")

    # 3. Duplicate movieId check
    dup_ids = df[df["movieId"].duplicated(keep=False)]
    if not dup_ids.empty:
        sample_dups = dup_ids["movieId"].unique()[:5].tolist()
        raise DataValidationError(
            f"Duplicate movie IDs detected: {len(sample_dups)} duplicate key(s) found (sample: {sample_dups}). "
            "Movie IDs must be unique primary keys."
        )

    # 4. Data type validation & casting
    try:
        df["movieId"] = df["movieId"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Column 'movieId' could not be parsed as integer IDs: {exc}") from exc

    df["title"] = df["title"].astype(str).str.strip()
    df["genres"] = df["genres"].fillna("(no genres listed)").astype(str).str.strip()

    # 5. Extract release year safely (e.g. 'Toy Story (1995)' -> 1995)
    # Never fabricate: if no 4-digit year in parentheses, leave as None
    year_pattern = re.compile(r"\((\d{4})\)\s*$")
    df["year"] = df["title"].apply(
        lambda t: int(m.group(1)) if (m := year_pattern.search(t)) else None
    )

    return df


def validate_ratings_dataframe(df: pd.DataFrame, valid_movie_ids: Optional[Set[int]] = None) -> pd.DataFrame:
    """
    Validates ratings dataset structure, range, and foreign key integrity.

    Checks:
    1. Required columns ('userId', 'movieId', 'rating', 'timestamp').
    2. Null checks across userId, movieId, and rating.
    3. Proper numeric types for ratings.
    4. Valid rating bounds (typically 0.5 to 5.0).
    5. Optionally checks whether ratings reference known movie IDs.

    Args:
        df: Raw DataFrame read from ratings.csv.
        valid_movie_ids: Optional set of movieIds from validated movies.csv.

    Returns:
        pd.DataFrame: Validated ratings DataFrame.

    Raises:
        DataValidationError: If ratings data is corrupted or invalid.
    """
    if df.empty:
        raise DataValidationError("The ratings dataset is empty.")

    missing_cols = REQUIRED_RATING_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Malformed ratings dataset. Missing required column(s): {sorted(list(missing_cols))}."
        )

    # Check for missing values
    for col in ["userId", "movieId", "rating"]:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            raise DataValidationError(f"Ratings dataset contains {null_count} null value(s) in column '{col}'.")

    # Type casting
    try:
        df["userId"] = df["userId"].astype("int64")
        df["movieId"] = df["movieId"].astype("int64")
        df["rating"] = df["rating"].astype(float)
        df["timestamp"] = df["timestamp"].astype("int64")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Failed to parse ratings data types: {exc}") from exc

    # Range check
    min_rating = df["rating"].min()
    max_rating = df["rating"].max()
    if min_rating < 0.0 or max_rating > 10.0:
        raise DataValidationError(
            f"Rating values out of expected bounds: found range [{min_rating}, {max_rating}]."
        )

    # Optional orphan rating check
    if valid_movie_ids is not None:
        orphan_ratings = df[~df["movieId"].isin(valid_movie_ids)]
        if not orphan_ratings.empty:
            orphan_count = len(orphan_ratings)
            orphan_movies = orphan_ratings["movieId"].nunique()
            logger.warning(
                "Dataset contains %d ratings referencing %d movie IDs not present in movies.csv.",
                orphan_count,
                orphan_movies,
            )

    return df


def validate_tags_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates tags dataset structure and cleans invalid entries.

    Args:
        df: Raw DataFrame from tags.csv.

    Returns:
        pd.DataFrame: Cleaned tags DataFrame.
    """
    if df.empty:
        return df

    missing_cols = REQUIRED_TAG_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Malformed tags dataset. Missing required column(s): {sorted(list(missing_cols))}."
        )

    # Drop null tags rather than fabricating
    df = df.dropna(subset=["userId", "movieId", "tag"]).copy()

    df["userId"] = df["userId"].astype("int64")
    df["movieId"] = df["movieId"].astype("int64")
    df["tag"] = df["tag"].astype(str).str.strip()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype("int64")

    return df


def validate_links_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates links dataset structure.

    Args:
        df: Raw DataFrame from links.csv.

    Returns:
        pd.DataFrame: Cleaned links DataFrame.
    """
    if df.empty:
        return df

    missing_cols = REQUIRED_LINK_COLUMNS - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Malformed links dataset. Missing required column(s): {sorted(list(missing_cols))}."
        )

    df["movieId"] = df["movieId"].astype("int64")
    df["imdbId"] = df["imdbId"].astype(str).str.zfill(7)
    if "tmdbId" in df.columns:
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Individual Loader Functions
# ---------------------------------------------------------------------------

def load_movies(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads and validates movies.csv from the given directory.

    Args:
        data_dir: Directory path containing the dataset files.

    Returns:
        pd.DataFrame: Validated movies DataFrame.

    Raises:
        DatasetNotFoundError: If movies.csv cannot be located.
        DataValidationError: If dataset is corrupt or invalid.
    """
    movies_path = os.path.join(data_dir, "movies.csv")
    if not os.path.isfile(movies_path):
        raise DatasetNotFoundError(
            f"Required file 'movies.csv' was not found in directory '{data_dir}'. "
            f"Please ensure the MovieLens dataset is placed in the '{data_dir}/' folder."
        )

    try:
        raw_df = pd.read_csv(movies_path, encoding="utf-8")
    except Exception as exc:
        raise DataValidationError(f"Could not read 'movies.csv': {exc}") from exc

    return validate_movies_dataframe(raw_df)


def load_ratings(data_dir: str = "data", valid_movie_ids: Optional[Set[int]] = None) -> pd.DataFrame:
    """
    Loads and validates ratings.csv from the given directory.

    Args:
        data_dir: Directory path containing the dataset files.
        valid_movie_ids: Optional set of movieIds from movies.csv.

    Returns:
        pd.DataFrame: Validated ratings DataFrame.

    Raises:
        DatasetNotFoundError: If ratings.csv cannot be located.
        DataValidationError: If dataset is corrupt or invalid.
    """
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not os.path.isfile(ratings_path):
        raise DatasetNotFoundError(
            f"Required file 'ratings.csv' was not found in directory '{data_dir}'. "
            f"Please ensure the MovieLens dataset is placed in the '{data_dir}/' folder."
        )

    try:
        raw_df = pd.read_csv(ratings_path, encoding="utf-8")
    except Exception as exc:
        raise DataValidationError(f"Could not read 'ratings.csv': {exc}") from exc

    return validate_ratings_dataframe(raw_df, valid_movie_ids=valid_movie_ids)


def load_tags(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Gracefully loads optional tags.csv if present.

    Args:
        data_dir: Directory path containing the dataset files.

    Returns:
        Optional[pd.DataFrame]: Validated tags DataFrame or None if file is missing.
    """
    tags_path = os.path.join(data_dir, "tags.csv")
    if not os.path.isfile(tags_path):
        logger.info("Optional file 'tags.csv' not found in '%s'. Skipping tags.", data_dir)
        return None

    try:
        raw_df = pd.read_csv(tags_path, encoding="utf-8")
        return validate_tags_dataframe(raw_df)
    except Exception as exc:
        logger.warning("Failed to load optional 'tags.csv': %s. Proceeding without tags.", exc)
        return None


def load_links(data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Gracefully loads optional links.csv if present.

    Args:
        data_dir: Directory path containing the dataset files.

    Returns:
        Optional[pd.DataFrame]: Validated links DataFrame or None if file is missing.
    """
    links_path = os.path.join(data_dir, "links.csv")
    if not os.path.isfile(links_path):
        logger.info("Optional file 'links.csv' not found in '%s'. Skipping links.", data_dir)
        return None

    try:
        raw_df = pd.read_csv(links_path, encoding="utf-8")
        return validate_links_dataframe(raw_df)
    except Exception as exc:
        logger.warning("Failed to load optional 'links.csv': %s. Proceeding without links.", exc)
        return None


# ---------------------------------------------------------------------------
# Composite Loader and Statistical Profiling
# ---------------------------------------------------------------------------

def compute_dataset_statistics(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
    links_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Computes summary metrics and distributions without fabricating data.

    Args:
        movies_df: Validated movies DataFrame.
        ratings_df: Validated ratings DataFrame.
        tags_df: Optional validated tags DataFrame.
        links_df: Optional validated links DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing core dataset metrics.
    """
    total_movies = len(movies_df)
    total_ratings = len(ratings_df)
    unique_users = int(ratings_df["userId"].nunique()) if total_ratings > 0 else 0
    unique_rated_movies = int(ratings_df["movieId"].nunique()) if total_ratings > 0 else 0

    # Rating statistics
    avg_rating = round(float(ratings_df["rating"].mean()), 2) if total_ratings > 0 else 0.0
    median_rating = round(float(ratings_df["rating"].median()), 2) if total_ratings > 0 else 0.0
    rating_std = round(float(ratings_df["rating"].std()), 2) if total_ratings > 0 else 0.0

    # Rating score distribution
    rating_dist = (
        ratings_df["rating"].value_counts().sort_index().to_dict() if total_ratings > 0 else {}
    )

    # Genre analysis
    all_genres = []
    for g_str in movies_df["genres"].dropna():
        for g in g_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                all_genres.append(g)

    genre_counts = pd.Series(all_genres).value_counts().to_dict() if all_genres else {}
    unique_genres_count = len(genre_counts)

    # Movies with no ratings
    rated_movie_ids = set(ratings_df["movieId"])
    all_movie_ids = set(movies_df["movieId"])
    unrated_count = len(all_movie_ids - rated_movie_ids)

    # Year statistics
    valid_years = movies_df["year"].dropna()
    year_min = int(valid_years.min()) if not valid_years.empty else None
    year_max = int(valid_years.max()) if not valid_years.empty else None

    # Tag statistics
    total_tags = len(tags_df) if tags_df is not None else 0
    unique_tags = int(tags_df["tag"].nunique()) if tags_df is not None and not tags_df.empty else 0

    return {
        "total_movies": total_movies,
        "total_ratings": total_ratings,
        "unique_users": unique_users,
        "unique_rated_movies": unique_rated_movies,
        "unrated_movies": unrated_count,
        "average_rating": avg_rating,
        "median_rating": median_rating,
        "rating_std": rating_std,
        "rating_distribution": rating_dist,
        "unique_genres_count": unique_genres_count,
        "genre_counts": genre_counts,
        "year_min": year_min,
        "year_max": year_max,
        "total_tags": total_tags,
        "unique_tags": unique_tags,
        "has_tags": tags_df is not None and not tags_df.empty,
        "has_links": links_df is not None and not links_df.empty,
    }


def enrich_movie_catalog(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Enriches the movies catalogue with actual aggregate statistics (rating count, average rating).
    Never fabricates synthetic reviews or metadata.

    Args:
        movies_df: Validated movies DataFrame.
        ratings_df: Validated ratings DataFrame.
        tags_df: Optional tags DataFrame.

    Returns:
        pd.DataFrame: Enriched catalogue DataFrame sorted by rating count.
    """
    # Group ratings by movieId
    rating_aggs = (
        ratings_df.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    rating_aggs["avg_rating"] = rating_aggs["avg_rating"].round(2)

    enriched = movies_df.merge(rating_aggs, on="movieId", how="left")
    enriched["rating_count"] = enriched["rating_count"].fillna(0).astype(int)
    enriched["avg_rating"] = enriched["avg_rating"].fillna(0.0)

    # Attach aggregated top tags if available
    if tags_df is not None and not tags_df.empty:
        tags_agg = (
            tags_df.groupby("movieId")["tag"]
            .apply(lambda tags: ", ".join(pd.Series(tags).value_counts().index[:3]))
            .reset_index()
            .rename(columns={"tag": "sample_tags"})
        )
        enriched = enriched.merge(tags_agg, on="movieId", how="left")
        enriched["sample_tags"] = enriched["sample_tags"].fillna("")
    else:
        enriched["sample_tags"] = ""

    return enriched


@cache_decorator(show_spinner=False)
def load_and_validate_dataset(
    data_dir: str = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Loads all datasets from the given directory with full schema validation.
    Equipped with Streamlit caching for performance when running in Streamlit.

    Args:
        data_dir: Path to directory containing MovieLens CSV files.

    Returns:
        Tuple of (movies_df, ratings_df, tags_df, links_df)
    """
    movies_df = load_movies(data_dir=data_dir)
    valid_movie_ids = set(movies_df["movieId"])
    ratings_df = load_ratings(data_dir=data_dir, valid_movie_ids=valid_movie_ids)
    tags_df = load_tags(data_dir=data_dir)
    links_df = load_links(data_dir=data_dir)
    return movies_df, ratings_df, tags_df, links_df

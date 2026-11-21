"""
fetch_posters.py
================
CineMatch AI — Movie Poster Ingestion CLI Tool
Downloads and verifies movie posters for the MovieLens dataset using the official TMDb API.

Usage:
    # 1. Inspect current manifest statistics:
    python fetch_posters.py --status

    # 2. Test run on a small batch of 10 movies:
    python fetch_posters.py --limit 10

    # 3. Process full MovieLens dataset with polite rate limiting:
    python fetch_posters.py

    # 4. Force re-download / refresh existing posters:
    python fetch_posters.py --limit 20 --overwrite
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

import pandas as pd

from data_loader import load_links, load_movies
from poster_service import (
    POSTERS_DIR,
    TMDB_ATTRIBUTION_TEXT,
    VERIFIED_TEST_MOVIES,
    PosterManifest,
    batch_fetch_posters,
    fetch_popular_posters_batch,
    fetch_verified_seed_batch,
    get_tmdb_api_key,
    is_api_key_configured,
)


# Reconfigure standard streams to UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner() -> None:
    print("=" * 74)
    print("CineMatch AI -- TMDb Poster Acquisition Pipeline")
    print(f"Attribution: {TMDB_ATTRIBUTION_TEXT}")
    print("=" * 74)


def print_api_key_help() -> None:
    print("\n[!] TMDB_API_KEY is not configured or is using a placeholder.")
    print("-" * 74)
    print("To obtain a free TMDb API key:")
    print("  1. Create a free account at: https://www.themoviedb.org/signup")
    print("  2. Verify your email and log in.")
    print("  3. Go to Account Settings > API (https://www.themoviedb.org/settings/api).")
    print("  4. Request an API Key (Developer / Educational) and copy your 'API Key (v3 auth)'.")
    print("  5. Create a file named '.env' in this project folder containing:")
    print("       TMDB_API_KEY=your_actual_api_key_here")
    print("  6. Re-run this script: python fetch_posters.py --limit 50")
    print("-" * 74 + "\n")


def format_status_symbol(status: str) -> str:
    if status == "downloaded":
        return "[OK:DOWNLOADED]"
    elif status == "no_poster":
        return "[--:NO POSTER ]"
    elif status == "not_found":
        return "[XX:NOT FOUND ]"
    elif status == "error":
        return "[!!:ERROR     ]"
    return f"[{status.upper()}]"


def run_seed_batch(manifest: PosterManifest, overwrite: bool = False) -> int:
    print("\n[+] Running verified 10-movie calibration & verification batch from TMDb CDN...")
    print("    Validating HTTP status codes (200), file sizes (>500B), and JPEG magic bytes.")
    start_time = time.time()

    def seed_callback(current: int, total: int, res: Dict[str, Any]) -> None:
        pct = (current / total) * 100 if total > 0 else 0
        st_sym = format_status_symbol(res.get("status", "unknown"))
        mid = res.get("movie_id")
        title = res.get("title", "")
        if len(title) > 36:
            title = title[:33] + "..."
        method = res.get("match_method", "none")
        lp = res.get("local_path") or res.get("error_message") or ""
        if len(lp) > 30:
            lp = lp[:27] + "..."
        print(f"[{current:2d}/{total:2d}] ({pct:5.1f}%) {st_sym} #{mid:<5d} {title:<36s} | {method:<14s} | {lp}")

    summary = fetch_verified_seed_batch(
        manifest=manifest,
        overwrite=overwrite,
        progress_callback=seed_callback,
    )

    elapsed = time.time() - start_time
    stats = manifest.get_summary_stats()

    print("\n" + "=" * 74)
    print("Verified Seed Batch Completed!")
    print(f"   Elapsed Time:     {elapsed:.1f} seconds")
    print(f"   Queued for Run:   {summary['total_queued']}")
    print(f"   Newly Downloaded: {summary['downloaded']}")
    print(f"   Already Cached:   {summary['existing']}")
    print(f"   Errors:           {summary['error']}")
    print("-" * 74)
    print(f"Archive Status:")
    print(f"   Total in Manifest:{stats['total_recorded']}")
    print(f"   Posters on Disk:  {stats['disk_verified']}")
    print("=" * 74)
    return 0 if summary["error"] == 0 else 1


def run_popular_batch(manifest: PosterManifest, limit: int = 60, data_dir: str = "data", overwrite: bool = False) -> int:
    print(f"\n[+] Running verified popular movie poster batch (up to {limit} movies)...")
    print("    Validating HTTP status codes (200), file sizes (>500B), and image magic bytes.")
    start_time = time.time()

    def batch_callback(current: int, total: int, res: Dict[str, Any]) -> None:
        pct = (current / total) * 100 if total > 0 else 0
        st_sym = format_status_symbol(res.get("status", "unknown"))
        mid = res.get("movie_id")
        title = res.get("title", "")
        if len(title) > 36:
            title = title[:33] + "..."
        method = res.get("match_method", "none")
        lp = res.get("local_path") or res.get("error_message") or ""
        if len(lp) > 30:
            lp = lp[:27] + "..."
        print(f"[{current:2d}/{total:2d}] ({pct:5.1f}%) {st_sym} #{mid:<5d} {title:<36s} | {method:<14s} | {lp}")

    summary = fetch_popular_posters_batch(
        data_dir=data_dir,
        manifest=manifest,
        limit=limit,
        overwrite=overwrite,
        progress_callback=batch_callback,
    )

    elapsed = time.time() - start_time
    stats = manifest.get_summary_stats()

    print("\n" + "=" * 74)
    print("Popular Movie Poster Batch Completed!")
    print(f"   Elapsed Time:     {elapsed:.1f} seconds")
    print(f"   Queued for Run:   {summary['total_queued']}")
    print(f"   Newly Downloaded: {summary['downloaded']}")
    print(f"   Already Cached:   {summary['existing']}")
    print(f"   Errors:           {summary['error']}")
    print("-" * 74)
    print(f"Archive Status:")
    print(f"   Total in Manifest:{stats['total_recorded']}")
    print(f"   Posters on Disk:  {stats['disk_verified']}")
    print("=" * 74)
    return 0 if summary["error"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch and cache authentic movie posters from TMDb for MovieLens movies."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing movies.csv and links.csv (default: 'data').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of movies to process in this run (e.g. 50 or 60).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Frequency of persisting manifest to disk (default: 20).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay in seconds between API requests to avoid rate limits (default: 0.25s).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional TMDb API key override. If not set, reads from .env or TMDB_API_KEY.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-fetching and downloading even if poster already exists on disk.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current poster download statistics from manifest and exit.",
    )
    parser.add_argument(
        "--seed-batch",
        action="store_true",
        help="Download and verify the initial verification batch directly from TMDb CDN.",
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Download and verify posters for popular MovieLens movies to reach 50+ posters.",
    )

    args = parser.parse_args()
    print_banner()

    manifest = PosterManifest()
    stats = manifest.get_summary_stats()

    if args.status:
        print("\n[i] Current Poster Manifest Statistics:")
        print(f"  * Total movies tracked in manifest: {stats['total_recorded']}")
        print(f"  * Posters verified on disk:         {stats['disk_verified']}")
        print(f"  * Status 'downloaded':              {stats['downloaded']}")
        print(f"  * Status 'no_poster' (TMDb empty):  {stats['no_poster']}")
        print(f"  * Status 'not_found' (No match):    {stats['not_found']}")
        print(f"  * Status 'error' (Network/API):     {stats['error']}")
        print(f"  * Poster storage folder:            {POSTERS_DIR}/\n")
        return 0

    # If --popular or --seed-batch requested explicitly
    if args.popular:
        return run_popular_batch(manifest, limit=args.limit or 60, data_dir=args.data_dir, overwrite=args.overwrite)
    if args.seed_batch:
        return run_seed_batch(manifest, overwrite=args.overwrite)

    # Verify API key
    active_key = get_tmdb_api_key(args.api_key)
    if not is_api_key_configured(active_key):
        # If user ran with a limit (e.g. 10, 50, 60), run popular batch to fulfill the quota
        if args.limit:
            print("\n[i] Notice: TMDB_API_KEY is not configured in .env.")
            print(f"    Running verified popular movie poster batch for {args.limit} films...")
            rc = run_popular_batch(manifest, limit=args.limit, data_dir=args.data_dir, overwrite=args.overwrite)
            print_api_key_help()
            return rc
        print_api_key_help()
        return 1


    # Load MovieLens dataset
    print(f"\n[+] Loading MovieLens dataset from '{args.data_dir}'...")
    try:
        movies_df = load_movies(args.data_dir)
        links_df = load_links(args.data_dir)
        print(f"   Loaded {len(movies_df):,} movies from movies.csv.")
        if links_df is not None:
            print(f"   Loaded {len(links_df):,} external identifiers from links.csv.")
        else:
            print("   [!] Optional links.csv not found. Falling back to title/year search matching.")

    except Exception as exc:
        print(f"[!] Failed to load dataset: {exc}")
        return 1

    print(f"\n[*] Configuration:")
    print(f"  * Limit:      {args.limit if args.limit else 'Full dataset'}")
    print(f"  * Rate delay: {args.delay}s per request")
    print(f"  * Overwrite:  {args.overwrite}")
    print(f"  * Directory:  {POSTERS_DIR}/\n")

    start_time = time.time()

    def progress_callback(current: int, total: int, res: Dict[str, Any]) -> None:
        pct = (current / total) * 100 if total > 0 else 0
        st_sym = format_status_symbol(res.get("status", "unknown"))
        mid = res.get("movie_id")
        title = res.get("title", "")
        # Truncate title for clean console alignment
        if len(title) > 36:
            title = title[:33] + "..."
        method = res.get("match_method", "none")
        lp = res.get("local_path") or res.get("error_message") or ""
        if len(lp) > 30:
            lp = lp[:27] + "..."
        print(f"[{current:4d}/{total:4d}] ({pct:5.1f}%) {st_sym} #{mid:<5d} {title:<36s} | {method:<14s} | {lp}")

    try:
        summary = batch_fetch_posters(
            movies_df=movies_df,
            links_df=links_df,
            api_key=active_key,
            limit=args.limit,
            batch_size=args.batch_size,
            delay=args.delay,
            overwrite=args.overwrite,
            progress_callback=progress_callback,
        )
    except KeyboardInterrupt:
        print("\n[!] Process interrupted by user (Ctrl+C). Partial progress was safely saved to manifest.")
        return 130
    except Exception as exc:
        print(f"\n[!] Error during poster ingestion: {exc}")
        return 1

    elapsed = time.time() - start_time
    updated_stats = manifest.get_summary_stats()

    print("\n" + "=" * 74)
    print("Poster Fetch Run Completed!")
    print(f"   Elapsed Time:     {elapsed:.1f} seconds")
    print(f"   Queued for Run:   {summary['total_queued']}")
    print(f"   Processed:        {summary['processed']}")
    print(f"   Downloaded (New): {summary['downloaded']}")
    print(f"   No Poster on TMDb:{summary['no_poster']}")
    print(f"   Not Found:        {summary['not_found']}")
    print(f"   Errors:           {summary['error']}")
    print("-" * 74)
    print(f"Archive Status:")
    print(f"   Total in Manifest:{updated_stats['total_recorded']}")
    print(f"   Posters on Disk:  {updated_stats['disk_verified']}")
    print("=" * 74)
    print("\nLaunch CineMatch AI to view your genuine movie posters:")
    print("   streamlit run app.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

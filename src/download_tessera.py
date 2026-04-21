import argparse
import os

from src.tessera_tiles import fetch_tessera_tiles
from src.tessera_points import fetch_tessera_points

DATASET_NAME = "global"
DEFAULT_TILE_SIZE = 128
DEFAULT_YEAR = 2024


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch TESSERA embeddings for specified dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="Root data directory (same as paths.data_dir in configs). Default: data/",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile size in pixels. Default: {DEFAULT_TILE_SIZE}",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Year to fetch embeddins for. Defualt: {DEFAULT_YEAR}"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Base directory for all TESSERA cache files. "
            "GeoTessera's registry is stored here; large raw source tiles go in "
            "the raw/ subfolder. "
            "Falls back to the TESSERA_EMBEDDINGS_DIR env var, then "
            "{data_dir}/cache/tessera. Set TESSERA_EMBEDDINGS_DIR in .env to "
            "avoid passing this flag every run."
        ),
    )
    parser.add_argument(
        "--points",
        action="store_true",
        default=False,
        help="Download embeddigns for center point only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel download threads. Default: 1. "
            "When writing to an external drive too many workers can cause I/O "
            "bottlenecks. Increase with caution."
        ),
    )
    parser.add_argument(
        "--retry-stuck",
        action="store_true",
        default=False,
        help="Clear stuck.txt and retry previously-stuck records instead of skipping them.",
    )
    args = parser.parse_args()

    if args.points:
        fetch_tessera_points(
            dataset=args.dataset,
            data_dir=args.data_dir,
            year=args.year,
            cache_dir=args.cache_dir,
            workers=args.workers,
            retry_stuck=args.retry_stuck,)
    else:
        fetch_tessera_tiles(
            dataset=args.dataset,
            data_dir=args.data_dir,
            tile_size=args.tile_size,
            year=args.year,
            cache_dir=args.cache_dir,
            workers=args.workers,
            retry_stuck=args.retry_stuck,
        )


if __name__ == "__main__":
    os.chdir('..')
    print(os.getcwd())
    main()
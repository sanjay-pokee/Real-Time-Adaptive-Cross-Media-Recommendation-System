"""Download one Amazon Reviews 2023 category file from HuggingFace.

Grabs the small "rating only" CSV (user_id, parent_asin, rating, timestamp)
for a category, which is all LightGCN training/eval needs. Falls back to the
raw review JSONL if no rating-only file is published for that category.

Usage:
    python -m scripts.fetch_amazon_dataset --category Movies_and_TV
    python -m scripts.fetch_amazon_dataset --category Books --variant 0core
    python -m scripts.fetch_amazon_dataset --category CDs_and_Vinyl --raw-review

The printed path is what you pass to `models.graph.train_lightgcn --dataset`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REPO_TYPE = "dataset"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "amazon"


def _pick_file(files: list[str], category: str, variant: str, raw_review: bool) -> str:
    cat = category.lower()
    if raw_review:
        wanted = [
            f for f in files
            if cat in f.lower() and "review" in f.lower() and "meta" not in f.lower()
            and f.lower().endswith((".jsonl", ".jsonl.gz", ".json.gz"))
        ]
        if not wanted:
            raise SystemExit(f"No raw review file found for category '{category}'.")
        return sorted(wanted, key=len)[0]

    # rating-only style CSVs, e.g. benchmark/5core/rating_only/Movies_and_TV.csv
    candidates = [
        f for f in files
        if cat in f.lower()
        and f.lower().endswith((".csv", ".csv.gz"))
        and "rating" in f.lower()
        and variant.lower() in f.lower()
        and "meta" not in f.lower()
    ]
    if not candidates:
        # looser: any csv for this category under the requested core variant
        candidates = [
            f for f in files
            if cat in f.lower()
            and f.lower().endswith((".csv", ".csv.gz"))
            and variant.lower() in f.lower()
        ]
    if not candidates:
        available = sorted({f for f in files if cat in f.lower()})
        listing = "\n  ".join(available[:40]) or "(none)"
        raise SystemExit(
            f"No '{variant}' rating file for '{category}'. Files seen for this category:\n  {listing}\n"
            "Try a different --variant (0core / 5core) or --raw-review."
        )
    return sorted(candidates, key=len)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch an Amazon Reviews 2023 category file.")
    parser.add_argument("--category", required=True, help="e.g. Movies_and_TV, Books, CDs_and_Vinyl")
    parser.add_argument("--variant", default="5core", help="5core (default) or 0core")
    parser.add_argument("--raw-review", action="store_true", help="download the full review JSONL instead")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Listing files in {REPO_ID} ...")
    files = HfApi().list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    remote = _pick_file(files, args.category, args.variant, args.raw_review)
    print(f"Selected: {remote}")

    args.out.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote,
        local_dir=str(args.out),
    )
    local_path = Path(local)
    print(f"\nDownloaded -> {local_path}")
    print(f"size: {local_path.stat().st_size / 1e6:.1f} MB")
    print("\nNext:")
    print(
        f"  .\\.venv\\Scripts\\python.exe -m models.graph.datasets "
        f"{local_path.relative_to(PROJECT_ROOT)} --user-core 10 --item-core 10 --max-users 60000"
    )


if __name__ == "__main__":
    main()

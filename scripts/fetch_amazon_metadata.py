"""Stream one Amazon Reviews 2023 *metadata* category into a catalog-ready CSV.

The rating-only files that ``scripts.fetch_amazon_dataset`` grabs carry no text,
so they can train LightGCN but cannot populate the content catalogue. The item
metadata lives in ``raw/meta_categories/meta_<Category>.jsonl`` instead.

Those files are large - Health_and_Household is 2.5 GB - and the catalogue only
needs a sample of them. So this streams the file over HTTP and stops as soon as
it has collected ``--limit`` usable records, which reads tens of megabytes
rather than gigabytes and needs no local disk for the original.

Usage:
    python -m scripts.fetch_amazon_metadata --category Industrial_and_Scientific
    python -m scripts.fetch_amazon_metadata --category Health_and_Household --limit 20000

The printed path is what you register under ``path:`` in config/datasets.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "datasets" / "amazon" / "meta"

OUTPUT_COLUMNS = [
    "parent_asin",
    "title",
    "description",
    "categories",
    "store",
    "average_rating",
    "rating_number",
    "main_category",
]


def _flatten(value: object) -> str:
    """Amazon ships description/features/categories as lists of strings."""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    if value is None:
        return ""
    return str(value).strip()


def _is_usable(record: dict) -> bool:
    """Keep rows that can actually be embedded and shown."""
    title = _flatten(record.get("title"))
    if len(title) < 3:
        return False
    # An item with no prose at all embeds to noise, so require some text.
    body = _flatten(record.get("description")) + _flatten(record.get("features"))
    return len(body) >= 20


def fetch_metadata(category: str, limit: int, out_dir: Path) -> Path:
    remote = "raw/meta_categories/meta_" + category + ".jsonl"
    url = hf_hub_url(REPO_ID, remote, repo_type="dataset")
    print("streaming " + remote)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (category + ".csv")

    kept = 0
    seen = 0
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    try:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                seen += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not _is_usable(record):
                    continue
                # description and features are both prose about the product;
                # concatenating them gives the embedder more to work with.
                body = " ".join(
                    part for part in (
                        _flatten(record.get("description")),
                        _flatten(record.get("features")),
                    ) if part
                )
                writer.writerow({
                    "parent_asin": _flatten(record.get("parent_asin")),
                    "title": _flatten(record.get("title")),
                    "description": body,
                    "categories": ", ".join(
                        str(c).strip() for c in (record.get("categories") or [])
                        if str(c).strip()
                    ),
                    "store": _flatten(record.get("store")),
                    "average_rating": record.get("average_rating") or "",
                    "rating_number": record.get("rating_number") or "",
                    "main_category": _flatten(record.get("main_category")),
                })
                kept += 1
                if kept % 2500 == 0:
                    print("  kept " + str(kept) + " of " + str(seen) + " scanned")
                if kept >= limit:
                    break
    finally:
        response.close()

    size_mb = out_path.stat().st_size / 1e6
    print("")
    print("wrote " + str(out_path))
    print("  rows: " + str(kept) + "  (scanned " + str(seen) + " records)")
    print("  size: " + format(size_mb, ".1f") + " MB")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream an Amazon Reviews 2023 metadata category to CSV."
    )
    parser.add_argument("--category", required=True,
                        help="e.g. Industrial_and_Scientific, Health_and_Household")
    parser.add_argument("--limit", type=int, default=20000,
                        help="stop after this many usable records (default 20000)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    fetch_metadata(args.category, args.limit, args.out)


if __name__ == "__main__":
    main()

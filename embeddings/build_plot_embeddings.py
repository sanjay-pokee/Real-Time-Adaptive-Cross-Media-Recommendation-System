"""Build the synopsis-only vectors used for plot queries (see backend/plot_index.py).

    python -m embeddings.build_plot_embeddings

Writes embeddings/plot_embeddings.npy and embeddings/plot_embedding_index.csv with
EMBEDDING_MODEL (the same encoder as the item vectors). About 2.5 minutes on an
RTX 3050 laptop GPU for 46k synopses with bge-small. Rebuild after the catalogue
or the model changes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.plot_index import synopsis_rows
from backend.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "content_catalog.csv"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--vectors", type=Path, default=Path(settings.plot_embeddings_path))
    parser.add_argument("--index", type=Path, default=Path(settings.plot_embedding_index_path))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    catalog = pd.read_csv(args.catalog, low_memory=False, usecols=["global_id", "content_type", "description"])
    rows = synopsis_rows(catalog)
    print(f"{len(rows):,} synopses to embed with {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    started = time.perf_counter()
    vectors = model.encode(
        rows["description"].astype(str).tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    args.vectors.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.vectors, vectors)
    rows[["global_id"]].to_csv(args.index, index=False)
    print(f"wrote {args.vectors} and {args.index} in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()

"""Prominence: how established an item is, in [0, 1], comparable across sources.

Pure meaning-based search has no notion of which items people actually know.
Over 120,747 rows, a query's nearest neighbours are mostly obscure titles whose
*names* echo the query ("Love Story", "My Family", "Dark Water"), while the
films a person means - Inception, The Godfather, Toy Story - sit hundreds or
thousands of places down. Every production recommender carries a popularity
prior for this reason; this is ours.

Sources measure popularity differently, so each is mapped to [0, 1] on its own:

* movie, show - TMDb ``vote_count`` (how many people rated it, all time), read
  from the raw TMDb exports when present. The catalogue's own ``popularity``
  column for TMDb is a *trending-this-week* figure and would favour whatever is
  new over what is established.
* music      - the 0-100 Spotify / Last.fm popularity index, used linearly. It
  is already a bounded index; a log scale would call nearly every song famous.
* book, health, industrial, finance - rating or review counts, log-scaled.

Counts are log-scaled and divided by the 99.9th percentile of their content
type, so one runaway best-seller does not flatten everything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# (content type, catalogue source, raw TMDb export) whose vote counts replace the
# catalogue's trending-popularity figure.
TMDB_SOURCES = (
    ("movie", "tmdb_api", "movies"),
    ("show", "tmdb_tv_api", "shows"),
)
BOUNDED_INDEX_TYPES = {"music"}


def compute_prominence(
    catalog: pd.DataFrame,
    tmdb_exports: dict[str, Path] | None = None,
) -> np.ndarray:
    """One prominence value per catalogue row, in row order."""
    n = len(catalog)
    if n == 0:
        return np.zeros(0)

    content_type = catalog.get("content_type", pd.Series("", index=catalog.index)).astype(str).to_numpy()
    source = catalog.get("source", pd.Series("", index=catalog.index)).astype(str).to_numpy()
    counts = (
        pd.to_numeric(catalog.get("popularity", pd.Series(0.0, index=catalog.index)), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
        .copy()
    )

    for ctype, src, export in TMDB_SOURCES:
        votes = _tmdb_vote_counts((tmdb_exports or {}).get(export))
        if votes is None:
            continue
        mask = (content_type == ctype) & (source == src)
        if not mask.any():
            continue
        ids = pd.to_numeric(catalog.loc[mask, "source_id"], errors="coerce")
        counts[mask] = ids.map(votes).fillna(0.0).to_numpy(dtype=float)

    prominence = np.zeros(n)
    for ctype in np.unique(content_type):
        mask = content_type == ctype
        if ctype in BOUNDED_INDEX_TYPES:
            prominence[mask] = np.clip(counts[mask] / 100.0, 0.0, 1.0)
            continue
        logged = np.log1p(counts[mask])
        ceiling = float(np.quantile(logged, 0.999)) if mask.sum() else 0.0
        if ceiling > 0:
            prominence[mask] = np.clip(logged / ceiling, 0.0, 1.0)
    return prominence


def _tmdb_vote_counts(path: Path | None) -> pd.Series | None:
    """TMDb id -> vote_count from a raw export, or None when it is not available."""
    if path is None or not Path(path).exists():
        return None
    try:
        raw = pd.read_csv(path, usecols=["id", "vote_count"])
    except (ValueError, OSError):
        return None
    raw = raw.dropna(subset=["id"]).drop_duplicates(subset=["id"])
    return pd.Series(
        pd.to_numeric(raw["vote_count"], errors="coerce").fillna(0.0).to_numpy(),
        index=pd.to_numeric(raw["id"], errors="coerce"),
    )

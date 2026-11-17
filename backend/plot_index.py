"""Synopsis-only vectors: a second view of films, shows and books for plot queries.

The main item vector embeds one string per item: title, genres, TMDb keywords,
synopsis and cast. Those lists drown the synopsis. Groundhog Day's synopsis
says "he finds himself repeating the same day over and over", yet for the
query "man relives the same day over and over" the film ranked 1,421st by the
main vector, behind songs and films whose titles echo the words. Embedding the
synopsis alone ranks it 110th, inside the candidate pool.

The index is small (46,354 synopses of at least 40 characters, 71 MB) and
searched exactly with one matrix product, so it needs no vector database.
Build it with ``python -m embeddings.build_plot_embeddings``; without the files
the recommender simply runs without it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PLOT_TYPES = ("movie", "show", "book")
MIN_SYNOPSIS_CHARS = 40


@dataclass
class PlotIndex:
    vectors: np.ndarray            # one L2-normalised row per synopsis
    global_ids: np.ndarray         # global_id per row
    row_of: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.vectors) != len(self.global_ids):
            raise ValueError(
                f"Plot index rows ({len(self.global_ids)}) do not match vector rows ({len(self.vectors)})."
            )
        if not self.row_of:
            self.row_of = {str(gid): row for row, gid in enumerate(self.global_ids)}

    @classmethod
    def load(cls, vectors_path: Path, index_path: Path) -> "PlotIndex | None":
        """The saved index, or None when it has not been built."""
        if not Path(vectors_path).exists() or not Path(index_path).exists():
            return None
        vectors = np.load(vectors_path).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, 1e-12)
        ids = pd.read_csv(index_path)["global_id"].astype(str).to_numpy(dtype=object)
        return cls(vectors, ids)

    def search(self, vector: np.ndarray, limit: int, allowed: np.ndarray | None = None) -> list[tuple[str, float]]:
        """(global_id, cosine) of the closest synopses, best first; ``allowed`` masks rows."""
        if limit <= 0 or not len(self.vectors):
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        limit = min(limit, len(scores))
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [(str(self.global_ids[row]), float(scores[row])) for row in top if np.isfinite(scores[row])]

    def cosines(self, vector: np.ndarray, global_ids: Iterable[str]) -> dict[str, float]:
        """Synopsis cosine for the items that have one."""
        rows = {gid: self.row_of[gid] for gid in global_ids if gid in self.row_of}
        if not rows:
            return {}
        block = self.vectors[np.fromiter(rows.values(), dtype=np.int64)]
        values = block @ np.asarray(vector, dtype=np.float32)
        return {gid: float(value) for gid, value in zip(rows, values)}


def synopsis_rows(catalog: pd.DataFrame) -> pd.DataFrame:
    """The catalogue rows that get a synopsis vector: films, shows and books with a real synopsis."""
    description = catalog.get("description", pd.Series("", index=catalog.index)).fillna("").astype(str)
    mask = catalog["content_type"].isin(PLOT_TYPES) & description.str.strip().str.len().ge(MIN_SYNOPSIS_CHARS)
    return catalog.loc[mask, ["global_id", "description"]]

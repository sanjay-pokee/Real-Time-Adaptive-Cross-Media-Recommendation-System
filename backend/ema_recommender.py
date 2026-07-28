"""Real-time EMA user profile vectors for instant personalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.graph.lightgcn import interaction_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings" / "content_embeddings.npy"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "embeddings" / "content_embedding_index.csv"


@dataclass
class EMAEmbeddingStore:
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH
    index_path: Path = DEFAULT_INDEX_PATH

    def __post_init__(self) -> None:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Content embeddings not found: {self.embeddings_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Embedding index not found: {self.index_path}")

        self.embeddings = _normalize_rows(np.load(self.embeddings_path).astype(np.float32))
        self.index = pd.read_csv(self.index_path)
        self.item_to_row = {
            str(row.global_id): int(row.embedding_row)
            for row in self.index.itertuples(index=False)
        }
        if len(self.index) != self.embeddings.shape[0]:
            raise ValueError(
                f"Embedding index rows ({len(self.index)}) do not match "
                f"embedding matrix rows ({self.embeddings.shape[0]})."
            )

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    def get_item_vector(self, global_id: str) -> np.ndarray | None:
        row = self.item_to_row.get(global_id)
        if row is None:
            return None
        return self.embeddings[row]

    def update_profile_vector(
        self,
        current_vector: list[float] | None,
        entity_id: str,
        event_type: str,
        event_value: float | None,
        alpha: float = 0.25,
    ) -> list[float] | None:
        item_vector = self.get_item_vector(entity_id)
        if item_vector is None:
            return current_vector

        signed_weight = interaction_weight(event_type, event_value)
        if signed_weight == 0:
            return current_vector

        direction = 1.0 if signed_weight > 0 else -1.0
        strength = min(abs(signed_weight), 1.0)
        step = float(np.clip(alpha * strength, 0.0, 1.0))
        target = direction * item_vector

        if not current_vector:
            updated = target
        else:
            current = np.asarray(current_vector, dtype=np.float32)
            if current.shape[0] != self.dimension:
                current = np.zeros(self.dimension, dtype=np.float32)
            updated = (1.0 - step) * current + step * target

        return _normalize_vector(updated).astype(float).tolist()

    def score(self, user_vector: list[float] | None, item_id: str) -> float | None:
        if not user_vector:
            return None
        item_vector = self.get_item_vector(item_id)
        if item_vector is None:
            return None
        user = np.asarray(user_vector, dtype=np.float32)
        if user.shape[0] != self.dimension:
            return None
        user = _normalize_vector(user)
        return float(user @ item_vector)

    def rerank(
        self,
        user_vector: list[float] | None,
        candidates: list[dict[str, Any]],
        ema_weight: float = 0.15,
    ) -> list[dict[str, Any]]:
        reranked = []
        for item in candidates:
            updated = dict(item)
            ema_score = self.score(user_vector, str(item.get("global_id", "")))
            updated["ema_score"] = ema_score
            if ema_score is not None:
                updated["score"] = float(updated.get("score", 0.0)) + ema_weight * ema_score
            reranked.append(updated)
        reranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return reranked


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return value.astype(np.float32)
    return (value / norm).astype(np.float32)

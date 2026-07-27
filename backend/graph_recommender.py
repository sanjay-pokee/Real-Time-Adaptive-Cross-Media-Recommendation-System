"""Load exported LightGCN embeddings and score candidate items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIGHTGCN_ARTIFACT = PROJECT_ROOT / "models" / "graph" / "artifacts" / "lightgcn_embeddings.npz"


@dataclass
class GraphEmbeddingStore:
    artifact_path: Path = DEFAULT_LIGHTGCN_ARTIFACT

    def __post_init__(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"LightGCN artifact not found: {self.artifact_path}")

        artifact = np.load(self.artifact_path, allow_pickle=True)
        self.user_ids = [str(value) for value in artifact["user_ids"].tolist()]
        self.item_ids = [str(value) for value in artifact["item_ids"].tolist()]
        self.user_embeddings = _normalize_rows(artifact["user_embeddings"].astype(np.float32))
        self.item_embeddings = _normalize_rows(artifact["item_embeddings"].astype(np.float32))
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.user_ids)}
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def score(self, user_id: str, item_id: str) -> float | None:
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        if user_idx is None or item_idx is None:
            return None
        return float(self.user_embeddings[user_idx] @ self.item_embeddings[item_idx])

    def rerank(
        self,
        user_id: str,
        candidates: list[dict],
        graph_weight: float = 0.2,
    ) -> list[dict]:
        reranked = []
        for item in candidates:
            graph_score = self.score(user_id, str(item.get("global_id", "")))
            updated = dict(item)
            semantic_score = float(updated.get("score", 0.0))
            updated["semantic_score"] = semantic_score
            updated["graph_score"] = graph_score
            if graph_score is not None:
                updated["score"] = semantic_score + graph_weight * graph_score
            reranked.append(updated)
        reranked.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return reranked


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)



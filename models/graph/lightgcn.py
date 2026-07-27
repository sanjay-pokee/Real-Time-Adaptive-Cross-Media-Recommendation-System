"""Minimal PyTorch LightGCN implementation for user-item recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


POSITIVE_EVENT_WEIGHTS = {
    "view": 0.4,
    "click": 0.6,
    "bookmark": 1.0,
    "like": 1.0,
    "complete": 1.2,
    "rating": 1.0,
}
NEGATIVE_EVENT_WEIGHTS = {
    "skip": -1.0,
}


@dataclass(frozen=True)
class LightGCNConfig:
    embedding_dim: int = 64
    num_layers: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 100
    batch_size: int = 1024
    seed: int = 42


class LightGCN(nn.Module):
    """LightGCN with symmetric normalized user-item graph propagation."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_users <= 0:
            raise ValueError("num_users must be greater than zero.")
        if num_items <= 0:
            raise ValueError("num_items must be greater than zero.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")
        if num_layers < 0:
            raise ValueError("num_layers cannot be negative.")

        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight],
            dim=0,
        )
        layer_embeddings = [embeddings]
        for _ in range(self.num_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)

        final_embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        users, items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return users, items

    def score_pairs(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        user_embeddings, item_embeddings = self.propagate(adjacency)
        return (user_embeddings[users] * item_embeddings[items]).sum(dim=1)


def interaction_weight(event_type: str, event_value: float | None = None) -> float:
    event = str(event_type or "").lower()
    if event == "rating":
        if event_value is None:
            return 1.0
        return max((float(event_value) - 3.0) / 2.0, -1.0)
    if event in POSITIVE_EVENT_WEIGHTS:
        return POSITIVE_EVENT_WEIGHTS[event]
    if event in NEGATIVE_EVENT_WEIGHTS:
        return NEGATIVE_EVENT_WEIGHTS[event]
    return 0.0


def build_training_interactions(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Return positive user-item pairs and stable id mappings."""
    required = {"user_id", "entity_id", "event_type"}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Interactions missing columns: {sorted(missing)}")

    if interactions.empty:
        raise ValueError("No interactions available for LightGCN training.")

    frame = interactions.copy()
    if "event_value" not in frame.columns:
        frame["event_value"] = None
    frame["weight"] = [
        interaction_weight(row.event_type, row.event_value)
        for row in frame.itertuples(index=False)
    ]
    frame = frame[frame["weight"] > 0].copy()
    if frame.empty:
        raise ValueError("No positive interactions available for LightGCN training.")

    user_ids = sorted(frame["user_id"].astype(str).unique())
    item_ids = sorted(frame["entity_id"].astype(str).unique())
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    frame["user_idx"] = frame["user_id"].astype(str).map(user_to_idx)
    frame["item_idx"] = frame["entity_id"].astype(str).map(item_to_idx)
    return frame[["user_id", "entity_id", "user_idx", "item_idx", "weight"]], user_to_idx, item_to_idx


def build_normalized_adjacency(
    user_indices: Iterable[int],
    item_indices: Iterable[int],
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    users = torch.as_tensor(list(user_indices), dtype=torch.long, device=device)
    items = torch.as_tensor(list(item_indices), dtype=torch.long, device=device) + num_users
    if users.numel() == 0:
        raise ValueError("Cannot build graph with zero edges.")

    rows = torch.cat([users, items])
    cols = torch.cat([items, users])
    indices = torch.stack([rows, cols])
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    node_count = num_users + num_items
    adjacency = torch.sparse_coo_tensor(indices, values, (node_count, node_count)).coalesce()

    degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp(min=1)
    row_degree = degree[adjacency.indices()[0]]
    col_degree = degree[adjacency.indices()[1]]
    norm_values = adjacency.values() / torch.sqrt(row_degree * col_degree)
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        norm_values,
        adjacency.shape,
        device=device,
    ).coalesce()


def bpr_loss(
    user_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
    reg_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight_decay: float,
) -> torch.Tensor:
    positive_scores = (user_embeddings * positive_embeddings).sum(dim=1)
    negative_scores = (user_embeddings * negative_embeddings).sum(dim=1)
    ranking_loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()
    regularization = sum(tensor.norm(2).pow(2) for tensor in reg_tensors) / user_embeddings.shape[0]
    return ranking_loss + weight_decay * regularization


def sample_bpr_batch(
    positive_pairs: np.ndarray,
    user_positive_items: dict[int, set[int]],
    num_items: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_rows = rng.integers(0, len(positive_pairs), size=batch_size)
    users = positive_pairs[batch_rows, 0]
    positive_items = positive_pairs[batch_rows, 1]
    negative_items = np.empty(batch_size, dtype=np.int64)

    for idx, user_idx in enumerate(users):
        positives = user_positive_items[int(user_idx)]
        negative = int(rng.integers(0, num_items))
        while negative in positives:
            negative = int(rng.integers(0, num_items))
        negative_items[idx] = negative

    return users.astype(np.int64), positive_items.astype(np.int64), negative_items


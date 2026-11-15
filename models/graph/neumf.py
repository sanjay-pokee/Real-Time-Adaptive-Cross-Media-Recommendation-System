"""Neural Collaborative Filtering (NeuMF; He et al., WWW 2017) as a baseline beside LightGCN.

The Phase-2 panel asked whether NCF's multi-layer perceptron over user-item
pairs would improve on our graph model. NeuMF fuses two towers:

    GMF:  p_u (.) q_i                       (element-wise product, linear interaction)
    MLP:  ReLU stack over [p'_u ; q'_i]     (non-linear interaction)
    y_ui = sigmoid(h^T [GMF ; MLP])

It is trained with the paper's pointwise log loss: each observed positive is
paired with ``negatives`` uniformly sampled unobserved items. Evaluation reuses
the LightGCN harness (same split, same masking, same metrics), so the two
numbers are directly comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from models.graph.lightgcn import build_positive_keys, build_training_interactions


@dataclass(frozen=True)
class NeuMFConfig:
    gmf_dim: int = 32
    mlp_dim: int = 64  # per side, so the MLP tower's input is 2 * mlp_dim
    mlp_layers: tuple[int, ...] = (64, 32, 16)
    negatives: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: int = 4096  # positives per step; each step also scores negatives * batch_size pairs
    seed: int = 42


class NeuMF(nn.Module):
    def __init__(self, num_users: int, num_items: int, config: NeuMFConfig) -> None:
        super().__init__()
        if not config.mlp_layers:
            raise ValueError("NeuMF needs at least one MLP layer.")
        self.gmf_user = nn.Embedding(num_users, config.gmf_dim)
        self.gmf_item = nn.Embedding(num_items, config.gmf_dim)
        self.mlp_user = nn.Embedding(num_users, config.mlp_dim)
        self.mlp_item = nn.Embedding(num_items, config.mlp_dim)

        layers: list[nn.Module] = []
        width = 2 * config.mlp_dim
        for out in config.mlp_layers:
            layers += [nn.Linear(width, out), nn.ReLU()]
            width = out
        self.mlp = nn.Sequential(*layers)
        self.predict = nn.Linear(config.gmf_dim + width, 1)
        self.gmf_dim = config.gmf_dim
        self.mlp_dim = config.mlp_dim

        for table in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(table.weight, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.predict.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.predict.bias)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Logits for aligned (user, item) pairs, shape ``(batch,)``."""
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
        return self.predict(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def score_all_items(self, users: torch.Tensor) -> torch.Tensor:
        """Logits for every item, shape ``(len(users), num_items)``.

        Exact, but cheaper than calling ``forward`` on every pair: the first MLP
        layer is linear in [p'_u ; q'_i], so it splits into a user part and an
        item part computed once, and the GMF half of the output layer reduces to
        a single matrix product.
        """
        first = self.mlp[0]
        weight_user = first.weight[:, : self.mlp_dim]
        weight_item = first.weight[:, self.mlp_dim :]
        user_part = self.mlp_user(users) @ weight_user.T + first.bias  # (B, h1)
        item_part = self.mlp_item.weight @ weight_item.T  # (I, h1)
        hidden = self.mlp[1:](user_part[:, None, :] + item_part[None, :, :])  # (B, I, h_last)

        out_weight = self.predict.weight[0]
        gmf_scores = (self.gmf_user(users) * out_weight[: self.gmf_dim]) @ self.gmf_item.weight.T
        mlp_scores = hidden @ out_weight[self.gmf_dim :]
        return gmf_scores + mlp_scores + self.predict.bias


def sample_negatives(
    users: np.ndarray,
    positive_keys: np.ndarray,
    num_items: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``count`` unobserved items per user (uniform, redrawn on collision).

    ``positive_keys`` is the sorted key array from ``build_positive_keys``.
    Returns aligned (users, negative_items) arrays of length ``len(users) * count``.
    """
    repeated = np.repeat(users.astype(np.int64), count)
    negatives = rng.integers(0, num_items, size=len(repeated)).astype(np.int64)
    for _ in range(64):
        keys = repeated * np.int64(num_items) + negatives
        pos = np.searchsorted(positive_keys, keys)
        np.clip(pos, 0, len(positive_keys) - 1, out=pos)
        collision = positive_keys[pos] == keys
        hits = int(collision.sum())
        if hits == 0:
            break
        negatives[collision] = rng.integers(0, num_items, size=hits)
    return repeated, negatives


def train_neumf(
    training_rows: pd.DataFrame,
    config: NeuMFConfig,
    device: torch.device | None = None,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> tuple[NeuMF, dict[str, int], dict[str, int], pd.DataFrame, list[float]]:
    """Train NeuMF on the positive rows. Returns the model, id maps, mapped rows and per-epoch loss."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training, user_to_idx, item_to_idx = build_training_interactions(training_rows)
    num_items = len(item_to_idx)
    if num_items < 2:
        raise ValueError("NeuMF needs at least two training items.")

    positive_pairs = training[["user_idx", "item_idx"]].drop_duplicates().to_numpy(dtype=np.int64)
    positive_keys = build_positive_keys(positive_pairs, num_items)
    model = NeuMF(len(user_to_idx), num_items, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()
    batch_size = min(config.batch_size, len(positive_pairs))
    steps = int(np.ceil(len(positive_pairs) / batch_size))
    log(
        f"NeuMF training: users={len(user_to_idx):,} items={num_items:,} "
        f"pairs={len(positive_pairs):,} negatives/positive={config.negatives} steps/epoch={steps} device={device}"
    )

    losses: list[float] = []
    clock = time.perf_counter()
    model.train()
    for epoch in range(1, config.epochs + 1):
        order = rng.permutation(len(positive_pairs))
        epoch_loss = 0.0
        for step in range(steps):
            batch = positive_pairs[order[step * batch_size : (step + 1) * batch_size]]
            neg_users, neg_items = sample_negatives(batch[:, 0], positive_keys, num_items, config.negatives, rng)
            users = np.concatenate([batch[:, 0], neg_users])
            items = np.concatenate([batch[:, 1], neg_items])
            labels = np.concatenate([np.ones(len(batch)), np.zeros(len(neg_users))]).astype(np.float32)

            logits = model(
                torch.as_tensor(users, device=device),
                torch.as_tensor(items, device=device),
            )
            loss = loss_fn(logits, torch.as_tensor(labels, device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        losses.append(epoch_loss / steps)
        if epoch == 1 or epoch == config.epochs or epoch % 5 == 0:
            elapsed = time.perf_counter() - clock
            log(f"epoch={epoch:03d}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed / 60:.1f}m")

    model.eval()
    return model, user_to_idx, item_to_idx, training, losses


def rank_top_k(
    model: NeuMF,
    user_indices: list[int],
    seen: list[set[int]],
    k: int,
    batch_size: int = 64,
) -> list[list[int]]:
    """Top-k unseen item indices per user, full-catalogue ranking with training items masked."""
    device = next(model.parameters()).device
    num_items = model.gmf_item.num_embeddings
    top_n = min(k, num_items)
    ranked: list[list[int]] = []
    for start in range(0, len(user_indices), batch_size):
        rows = user_indices[start : start + batch_size]
        scores = model.score_all_items(torch.as_tensor(rows, dtype=torch.long, device=device))
        mask_rows: list[int] = []
        mask_cols: list[int] = []
        for row_pos, items in enumerate(seen[start : start + batch_size]):
            mask_cols.extend(items)
            mask_rows.extend([row_pos] * len(items))
        if mask_rows:
            scores[
                torch.as_tensor(mask_rows, dtype=torch.long, device=device),
                torch.as_tensor(mask_cols, dtype=torch.long, device=device),
            ] = float("-inf")
        ranked.extend(torch.topk(scores, top_n, dim=1).indices.cpu().tolist())
    return ranked

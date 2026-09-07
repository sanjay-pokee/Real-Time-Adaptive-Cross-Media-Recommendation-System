"""Evaluate LightGCN with holdout ranking metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.graph.lightgcn import (
    LightGCN,
    LightGCNConfig,
    bpr_loss,
    build_normalized_adjacency,
    build_positive_keys,
    build_training_interactions,
    interaction_weight,
    sample_bpr_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RankingMetrics:
    users_evaluated: int
    hit_rate: float
    recall: float
    precision: float
    ndcg: float
    mrr: float


def split_holdout_by_user(
    interactions: pd.DataFrame,
    holdout_per_user: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = _positive_interactions(interactions)
    if "timestamp" in positives.columns:
        positives = positives.sort_values(["user_id", "timestamp"])
    else:
        positives = positives.sort_values(["user_id", "entity_id"])

    train_parts = []
    test_parts = []
    for _, group in positives.groupby("user_id", sort=True):
        group = group.drop_duplicates(subset=["entity_id"], keep="last")
        if len(group) <= holdout_per_user:
            train_parts.append(group)
            continue
        test_parts.append(group.tail(holdout_per_user))
        train_parts.append(group.iloc[:-holdout_per_user])

    if not train_parts or not test_parts:
        raise ValueError("Not enough positive interactions per user for holdout evaluation.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def train_for_evaluation(
    training_rows: pd.DataFrame,
    config: LightGCNConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DataFrame]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training, user_to_idx, item_to_idx = build_training_interactions(training_rows)
    if len(item_to_idx) < 2:
        raise ValueError("LightGCN evaluation needs at least two training items.")

    adjacency = build_normalized_adjacency(
        training["user_idx"].to_numpy(),
        training["item_idx"].to_numpy(),
        len(user_to_idx),
        len(item_to_idx),
        device,
    )
    model = LightGCN(
        num_users=len(user_to_idx),
        num_items=len(item_to_idx),
        embedding_dim=config.embedding_dim,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.0)

    positive_pairs = training[["user_idx", "item_idx"]].drop_duplicates().to_numpy(dtype=np.int64)
    positive_keys = build_positive_keys(positive_pairs, len(item_to_idx))
    steps_per_epoch = max(1, int(np.ceil(len(positive_pairs) / config.batch_size)))

    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            users, positive_items, negative_items = sample_bpr_batch(
                positive_pairs,
                positive_keys,
                len(item_to_idx),
                min(config.batch_size, len(positive_pairs)),
                rng,
            )
            user_tensor = torch.as_tensor(users, dtype=torch.long, device=device)
            positive_tensor = torch.as_tensor(positive_items, dtype=torch.long, device=device)
            negative_tensor = torch.as_tensor(negative_items, dtype=torch.long, device=device)

            propagated_users, propagated_items = model.propagate(adjacency)
            loss = bpr_loss(
                propagated_users[user_tensor],
                propagated_items[positive_tensor],
                propagated_items[negative_tensor],
                (
                    model.user_embedding(user_tensor),
                    model.item_embedding(positive_tensor),
                    model.item_embedding(negative_tensor),
                ),
                config.weight_decay,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())

        if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
            print(f"epoch={epoch:03d} eval_train_loss={epoch_loss / steps_per_epoch:.4f}")

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)

    idx_to_user = sorted(user_to_idx, key=user_to_idx.get)
    idx_to_item = sorted(item_to_idx, key=item_to_idx.get)
    return (
        _normalize_rows(user_embeddings.detach().cpu().numpy().astype(np.float32)),
        _normalize_rows(item_embeddings.detach().cpu().numpy().astype(np.float32)),
        idx_to_user,
        idx_to_item,
        training,
    )


def _seen_index_sets(rows: pd.DataFrame, item_to_idx: dict[str, int]) -> dict[str, set[int]]:
    return (
        rows.groupby("user_id")["entity_id"]
        .apply(lambda values: {item_to_idx[str(v)] for v in values if str(v) in item_to_idx})
        .to_dict()
    )


def _metrics_from_rankings(
    ranked: dict[str, list[int]],
    test_relevant: dict[str, set[int]],
    k: int,
) -> RankingMetrics:
    hit_rates: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []

    for user_id, ranked_idx in ranked.items():
        relevant = test_relevant.get(user_id) or set()
        if not relevant:
            continue
        hits = [1 if idx in relevant else 0 for idx in ranked_idx[:k]]
        hit_count = sum(hits)
        hit_rates.append(1.0 if hit_count else 0.0)
        recalls.append(hit_count / len(relevant))
        precisions.append(hit_count / k)
        ndcgs.append(_ndcg_at_k(hits, min(k, len(relevant))))
        mrrs.append(_reciprocal_rank(hits))

    if not hit_rates:
        raise ValueError("No users could be evaluated. Check train/test overlap and item coverage.")

    return RankingMetrics(
        users_evaluated=len(hit_rates),
        hit_rate=float(np.mean(hit_rates)),
        recall=float(np.mean(recalls)),
        precision=float(np.mean(precisions)),
        ndcg=float(np.mean(ndcgs)),
        mrr=float(np.mean(mrrs)),
    )


def evaluate_rankings(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    user_ids: list[str],
    item_ids: list[str],
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    k: int = 10,
    eval_batch_size: int = 2048,
) -> RankingMetrics:
    """Batched full-catalog ranking evaluation (GPU when available)."""
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    train_seen = _seen_index_sets(train_rows, item_to_idx)
    test_relevant = _seen_index_sets(test_rows, item_to_idx)

    eval_users = [u for u, rel in test_relevant.items() if rel and u in user_to_idx]
    if not eval_users:
        raise ValueError("No users could be evaluated. Check train/test overlap and item coverage.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_tensor = torch.as_tensor(np.asarray(user_embeddings), dtype=torch.float32, device=device)
    item_tensor_t = torch.as_tensor(
        np.asarray(item_embeddings), dtype=torch.float32, device=device
    ).t().contiguous()
    num_items = item_tensor_t.shape[1]
    top_n = min(k, num_items)

    ranked: dict[str, list[int]] = {}
    for start in range(0, len(eval_users), eval_batch_size):
        batch = eval_users[start : start + eval_batch_size]
        rows = [user_to_idx[u] for u in batch]
        try:
            scores = (user_tensor[rows] @ item_tensor_t).clone()
        except RuntimeError:  # e.g. CUDA OOM -> retry on CPU
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            user_tensor = user_tensor.cpu()
            item_tensor_t = item_tensor_t.cpu()
            scores = (user_tensor[rows] @ item_tensor_t).clone()

        for row_pos, user_id in enumerate(batch):
            seen = train_seen.get(user_id)
            if seen:
                scores[row_pos, list(seen)] = float("-inf")

        top_indices = torch.topk(scores, top_n, dim=1).indices.cpu().numpy()
        for row_pos, user_id in enumerate(batch):
            ranked[user_id] = top_indices[row_pos].tolist()

    return _metrics_from_rankings(ranked, test_relevant, k)


def popularity_baseline(
    item_ids: list[str],
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    k: int = 10,
) -> RankingMetrics:
    """Most-popular recommender: rank every user by global train popularity."""
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    counts = train_rows["entity_id"].astype(str).map(item_to_idx).dropna().astype(int)
    popularity = np.zeros(len(item_ids), dtype=np.float64)
    for idx, count in counts.value_counts().items():
        popularity[int(idx)] = float(count)
    global_order = np.argsort(-popularity)

    train_seen = _seen_index_sets(train_rows, item_to_idx)
    test_relevant = _seen_index_sets(test_rows, item_to_idx)
    max_seen = max((len(s) for s in train_seen.values()), default=0)
    shortlist = global_order[: k + max_seen + 1]

    ranked: dict[str, list[int]] = {}
    for user_id, relevant in test_relevant.items():
        if not relevant:
            continue
        seen = train_seen.get(user_id, set())
        picks = [int(idx) for idx in shortlist if idx not in seen][:k]
        ranked[user_id] = picks
    return _metrics_from_rankings(ranked, test_relevant, k)


def _print_metrics(title: str, k: int, metrics: RankingMetrics) -> None:
    print(f"\n{title}")
    print(f"users_evaluated: {metrics.users_evaluated}")
    print(f"HitRate@{k}:   {metrics.hit_rate:.4f}")
    print(f"Recall@{k}:    {metrics.recall:.4f}")
    print(f"Precision@{k}: {metrics.precision:.4f}")
    print(f"NDCG@{k}:      {metrics.ndcg:.4f}")
    print(f"MRR@{k}:       {metrics.mrr:.4f}")


def run_evaluation(
    k: int,
    holdout_per_user: int,
    config: LightGCNConfig,
    interactions: pd.DataFrame | None = None,
    baseline: str = "none",
    eval_batch_size: int = 2048,
) -> RankingMetrics:
    if interactions is None:
        from backend.mysql_store import MySQLStore

        interactions = MySQLStore().get_lightgcn_interactions()
    train_rows, test_rows = split_holdout_by_user(interactions, holdout_per_user=holdout_per_user)

    if baseline == "popularity":
        item_ids = sorted(train_rows["entity_id"].astype(str).unique())
        metrics = popularity_baseline(item_ids, train_rows, test_rows, k=k)
        _print_metrics("Most-popular baseline", k, metrics)
        return metrics

    user_embeddings, item_embeddings, user_ids, item_ids, mapped_train = train_for_evaluation(train_rows, config)
    metrics = evaluate_rankings(
        user_embeddings,
        item_embeddings,
        user_ids,
        item_ids,
        mapped_train,
        test_rows,
        k=k,
        eval_batch_size=eval_batch_size,
    )
    _print_metrics("LightGCN holdout evaluation", k, metrics)
    return metrics


def _positive_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    frame = interactions.copy()
    if "event_value" not in frame.columns:
        frame["event_value"] = None
    frame["weight"] = [
        interaction_weight(row.event_type, row.event_value)
        for row in frame.itertuples(index=False)
    ]
    return frame[frame["weight"] > 0].copy()


def _ndcg_at_k(hits: list[int], ideal_hits: int) -> float:
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    if ideal == 0:
        return 0.0
    return float(dcg / ideal)


def _reciprocal_rank(hits: list[int]) -> float:
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LightGCN with holdout ranking metrics.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--holdout-per-user", type=int, default=1)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument(
        "--baseline",
        choices=["none", "popularity"],
        default="none",
        help="Run a non-learned baseline instead of training LightGCN.",
    )
    # Benchmark-dataset source (omit to evaluate from MySQL logs).
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--user-core", type=int, default=10)
    parser.add_argument("--item-core", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0, help="0 = keep all users")
    args = parser.parse_args()

    config = LightGCNConfig(
        embedding_dim=args.embedding_dim,
        num_layers=args.layers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    interactions = None
    if args.dataset:
        from models.graph.datasets import describe, prepare_interactions

        interactions = prepare_interactions(
            args.dataset,
            min_rating=args.min_rating,
            user_core=args.user_core,
            item_core=args.item_core,
            max_users=args.max_users,
            seed=args.seed,
        )
        print(f"dataset: {args.dataset}")
        print("  " + describe(interactions))

    run_evaluation(
        args.k,
        args.holdout_per_user,
        config,
        interactions=interactions,
        baseline=args.baseline,
        eval_batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    main()

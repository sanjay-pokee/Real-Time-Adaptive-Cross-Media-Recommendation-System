"""Evaluate NeuMF on the same holdout protocol as models.graph.evaluate_lightgcn.

Same dataset preparation, same leave-last-one-out split, same full-catalogue
ranking with training items masked, same metrics - so NeuMF, LightGCN and the
popularity baseline land in one comparable table.

    python -m models.graph.evaluate_neumf \
        --dataset datasets/amazon/benchmark/5core/rating_only/Industrial_and_Scientific.csv \
        --user-core 5 --item-core 5 --epochs 20 --k 20 \
        --report reports/benchmarks/industrial_and_scientific_neumf.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from models.graph.evaluate_lightgcn import (
    RankingMetrics,
    _metrics_from_rankings,
    _print_metrics,
    _seen_index_sets,
    split_holdout_by_user,
    write_report,
)
from models.graph.neumf import NeuMFConfig, rank_top_k, train_neumf


def run_neumf_evaluation(
    interactions: pd.DataFrame,
    config: NeuMFConfig,
    k: int = 20,
    holdout_per_user: int = 1,
    eval_batch_size: int = 64,
    report_path: Path | None = None,
    dataset_name: str = "interactions",
) -> RankingMetrics:
    train_rows, test_rows = split_holdout_by_user(interactions, holdout_per_user=holdout_per_user)
    model, user_to_idx, item_to_idx, mapped_train, losses = train_neumf(train_rows, config)

    train_seen = _seen_index_sets(mapped_train, item_to_idx)
    test_relevant = _seen_index_sets(test_rows, item_to_idx)
    eval_users = [u for u, relevant in test_relevant.items() if relevant and u in user_to_idx]

    clock = time.perf_counter()
    top = rank_top_k(
        model,
        [user_to_idx[u] for u in eval_users],
        [train_seen.get(u, set()) for u in eval_users],
        k,
        batch_size=eval_batch_size,
    )
    print(f"ranked {len(eval_users):,} users in {time.perf_counter() - clock:.1f}s", flush=True)
    metrics = _metrics_from_rankings(dict(zip(eval_users, top)), test_relevant, k)
    _print_metrics("NeuMF holdout evaluation", k, metrics)

    if report_path:
        write_report(
            report_path,
            metrics,
            k,
            model="neumf",
            dataset=dataset_name,
            config=config,
            graph={
                "users": len(user_to_idx),
                "items": len(item_to_idx),
                "interactions": int(len(mapped_train)),
                "final_train_loss": round(losses[-1], 6),
            },
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NeuMF with the LightGCN holdout protocol.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--user-core", type=int, default=10)
    parser.add_argument("--item-core", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0, help="0 = keep all users")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--holdout-per-user", type=int, default=1)
    parser.add_argument("--gmf-dim", type=int, default=32)
    parser.add_argument("--mlp-dim", type=int, default=64)
    parser.add_argument("--mlp-layers", type=str, default="64,32,16")
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

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

    config = NeuMFConfig(
        gmf_dim=args.gmf_dim,
        mlp_dim=args.mlp_dim,
        mlp_layers=tuple(int(width) for width in args.mlp_layers.split(",") if width.strip()),
        negatives=args.negatives,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_neumf_evaluation(
        interactions,
        config,
        k=args.k,
        holdout_per_user=args.holdout_per_user,
        eval_batch_size=args.eval_batch_size,
        report_path=args.report,
        dataset_name=str(args.dataset),
    )


if __name__ == "__main__":
    main()

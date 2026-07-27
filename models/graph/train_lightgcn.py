"""Train LightGCN from MySQL interaction logs and export embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from backend.mysql_store import MySQLStore
from models.graph.lightgcn import (
    LightGCN,
    LightGCNConfig,
    bpr_loss,
    build_normalized_adjacency,
    build_training_interactions,
    sample_bpr_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "graph" / "artifacts" / "lightgcn_embeddings.npz"


def train(output_path: Path, config: LightGCNConfig) -> Path:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interactions = MySQLStore().get_lightgcn_interactions()
    training, user_to_idx, item_to_idx = build_training_interactions(interactions)
    if len(item_to_idx) < 2:
        raise ValueError("LightGCN needs at least two positively interacted items.")

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    positive_pairs = training[["user_idx", "item_idx"]].drop_duplicates().to_numpy(dtype=np.int64)
    user_positive_items = (
        training.groupby("user_idx")["item_idx"]
        .apply(lambda values: set(int(value) for value in values))
        .to_dict()
    )
    steps_per_epoch = max(1, int(np.ceil(len(positive_pairs) / config.batch_size)))

    model.train()
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            users, positive_items, negative_items = sample_bpr_batch(
                positive_pairs,
                user_positive_items,
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
            print(f"epoch={epoch:03d} loss={epoch_loss / steps_per_epoch:.4f}")

    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx_to_user = np.array(sorted(user_to_idx, key=user_to_idx.get), dtype=object)
    idx_to_item = np.array(sorted(item_to_idx, key=item_to_idx.get), dtype=object)
    np.savez_compressed(
        output_path,
        user_ids=idx_to_user,
        item_ids=idx_to_item,
        user_embeddings=user_embeddings.detach().cpu().numpy().astype(np.float32),
        item_embeddings=item_embeddings.detach().cpu().numpy().astype(np.float32),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_users": len(user_to_idx),
                "num_items": len(item_to_idx),
                "num_positive_interactions": int(len(training)),
                "config": config.__dict__,
                "artifact": str(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LightGCN embeddings: {output_path}")
    print(f"Saved LightGCN metadata: {metadata_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGCN collaborative embeddings.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
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
    train(args.output, config)


if __name__ == "__main__":
    main()


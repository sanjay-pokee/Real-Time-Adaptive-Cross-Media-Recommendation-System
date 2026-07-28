import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ema_recommender import EMAEmbeddingStore
from backend.graph_recommender import GraphEmbeddingStore
from models.graph.evaluate_lightgcn import evaluate_rankings, split_holdout_by_user
from models.graph.lightgcn import (
    LightGCN,
    build_normalized_adjacency,
    build_training_interactions,
    interaction_weight,
    sample_bpr_batch,
)


def test_interaction_weight_maps_events():
    assert interaction_weight("like") > 0
    assert interaction_weight("bookmark") > interaction_weight("view")
    assert interaction_weight("skip") < 0
    assert interaction_weight("rating", 5) > 0
    assert interaction_weight("rating", 1) < 0


def test_build_training_interactions_keeps_positive_events_only():
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "entity_id": "i1", "event_type": "like", "event_value": 1},
            {"user_id": "u1", "entity_id": "i2", "event_type": "skip", "event_value": 1},
            {"user_id": "u2", "entity_id": "i1", "event_type": "rating", "event_value": 5},
        ]
    )

    training, user_to_idx, item_to_idx = build_training_interactions(interactions)

    assert len(training) == 2
    assert user_to_idx == {"u1": 0, "u2": 1}
    assert item_to_idx == {"i1": 0}


def test_build_normalized_adjacency_shape():
    adjacency = build_normalized_adjacency([0, 1], [0, 1], 2, 2, torch.device("cpu"))

    assert adjacency.shape == (4, 4)
    assert adjacency._nnz() == 4


def test_lightgcn_propagate_shapes():
    model = LightGCN(num_users=2, num_items=3, embedding_dim=8, num_layers=1)
    adjacency = build_normalized_adjacency([0, 1], [0, 1], 2, 3, torch.device("cpu"))

    users, items = model.propagate(adjacency)

    assert users.shape == (2, 8)
    assert items.shape == (3, 8)


def test_sample_bpr_batch_never_samples_known_positive():
    pairs = np.array([[0, 0], [0, 1], [1, 2]], dtype=np.int64)
    positives = {0: {0, 1}, 1: {2}}
    rng = np.random.default_rng(42)

    users, _, negatives = sample_bpr_batch(pairs, positives, 3, 20, rng)

    for user_idx, negative_idx in zip(users, negatives):
        assert int(negative_idx) not in positives[int(user_idx)]


def test_graph_embedding_store_scores_and_reranks(tmp_path):
    artifact_path = tmp_path / "lightgcn_embeddings.npz"
    np.savez_compressed(
        artifact_path,
        user_ids=np.array(["u1"], dtype=object),
        item_ids=np.array(["i1", "i2"], dtype=object),
        user_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        item_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    store = GraphEmbeddingStore(artifact_path)

    assert store.score("u1", "i1") == pytest.approx(1.0)
    assert store.score("u1", "missing") is None
    reranked = store.rerank(
        "u1",
        [{"global_id": "i2", "score": 0.9}, {"global_id": "i1", "score": 0.8}],
        graph_weight=0.2,
    )
    assert reranked[0]["global_id"] == "i1"
    assert reranked[0]["graph_score"] == pytest.approx(1.0)



def test_split_holdout_by_user_keeps_last_positive_per_user():
    interactions = pd.DataFrame(
        [
            {"user_id": "u1", "entity_id": "i1", "event_type": "like", "event_value": 1, "timestamp": 1},
            {"user_id": "u1", "entity_id": "i2", "event_type": "like", "event_value": 1, "timestamp": 2},
            {"user_id": "u1", "entity_id": "i3", "event_type": "like", "event_value": 1, "timestamp": 3},
            {"user_id": "u2", "entity_id": "i1", "event_type": "like", "event_value": 1, "timestamp": 1},
            {"user_id": "u2", "entity_id": "i4", "event_type": "skip", "event_value": 1, "timestamp": 2},
            {"user_id": "u2", "entity_id": "i5", "event_type": "like", "event_value": 1, "timestamp": 3},
            {"user_id": "u2", "entity_id": "i6", "event_type": "like", "event_value": 1, "timestamp": 4},
        ]
    )

    train, test = split_holdout_by_user(interactions, holdout_per_user=1)

    assert set(test["entity_id"]) == {"i3", "i6"}
    assert "i4" not in set(train["entity_id"])


def test_evaluate_rankings_calculates_metrics():
    user_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    item_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.0],
            [0.1, 0.0],
        ],
        dtype=np.float32,
    )
    train_rows = pd.DataFrame(
        [{"user_id": "u1", "entity_id": "i1", "user_idx": 0, "item_idx": 0, "weight": 1.0}]
    )
    test_rows = pd.DataFrame(
        [{"user_id": "u1", "entity_id": "i2", "event_type": "like", "event_value": 1}]
    )

    metrics = evaluate_rankings(
        user_embeddings,
        item_embeddings,
        ["u1"],
        ["i1", "i2", "i3"],
        train_rows,
        test_rows,
        k=2,
    )

    assert metrics.users_evaluated == 1
    assert metrics.hit_rate == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.ndcg == pytest.approx(1.0)
    assert metrics.mrr == pytest.approx(1.0)



def test_ema_embedding_store_updates_and_reranks(tmp_path):
    embeddings_path = tmp_path / "content_embeddings.npy"
    index_path = tmp_path / "content_embedding_index.csv"
    np.save(
        embeddings_path,
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {"global_id": "i1", "embedding_row": 0},
            {"global_id": "i2", "embedding_row": 1},
        ]
    ).to_csv(index_path, index=False)
    store = EMAEmbeddingStore(embeddings_path, index_path)

    vector = store.update_profile_vector([], "i1", "like", 1, alpha=0.5)
    assert vector is not None
    assert store.score(vector, "i1") == pytest.approx(1.0)

    reranked = store.rerank(
        vector,
        [{"global_id": "i2", "score": 0.9}, {"global_id": "i1", "score": 0.8}],
        ema_weight=0.2,
    )
    assert reranked[0]["global_id"] == "i1"
    assert reranked[0]["ema_score"] == pytest.approx(1.0)


def test_ema_embedding_store_negative_interaction_moves_away(tmp_path):
    embeddings_path = tmp_path / "content_embeddings.npy"
    index_path = tmp_path / "content_embedding_index.csv"
    np.save(embeddings_path, np.array([[1.0, 0.0]], dtype=np.float32))
    pd.DataFrame([{"global_id": "i1", "embedding_row": 0}]).to_csv(index_path, index=False)
    store = EMAEmbeddingStore(embeddings_path, index_path)

    vector = store.update_profile_vector([], "i1", "skip", 1, alpha=1.0)

    assert store.score(vector, "i1") == pytest.approx(-1.0)

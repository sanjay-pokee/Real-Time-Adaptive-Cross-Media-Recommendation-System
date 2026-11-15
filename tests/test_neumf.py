import numpy as np
import pandas as pd
import torch

from models.graph.lightgcn import build_positive_keys
from models.graph.neumf import NeuMF, NeuMFConfig, rank_top_k, sample_negatives, train_neumf


def _tiny_config(**overrides) -> NeuMFConfig:
    values = dict(gmf_dim=4, mlp_dim=6, mlp_layers=(8, 4), epochs=30, batch_size=16, learning_rate=0.01, seed=7)
    values.update(overrides)
    return NeuMFConfig(**values)


def test_score_all_items_matches_pairwise_forward():
    torch.manual_seed(0)
    model = NeuMF(num_users=5, num_items=9, config=_tiny_config()).eval()
    for parameter in model.parameters():  # move off the tiny init so every term matters
        torch.nn.init.normal_(parameter, std=0.5)
    users = torch.tensor([0, 3, 4])

    fast = model.score_all_items(users)
    with torch.no_grad():
        slow = torch.stack(
            [model(torch.full((9,), int(u)), torch.arange(9)) for u in users]
        )
    assert fast.shape == (3, 9)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_sample_negatives_never_returns_an_observed_item():
    pairs = np.array([[0, 0], [0, 1], [0, 2], [1, 3]], dtype=np.int64)
    keys = build_positive_keys(pairs, num_items=5)
    users, negatives = sample_negatives(np.array([0, 0, 1]), keys, 5, count=50, rng=np.random.default_rng(1))
    assert len(users) == len(negatives) == 150
    observed = {(0, 0), (0, 1), (0, 2), (1, 3)}
    assert not any((int(u), int(i)) in observed for u, i in zip(users, negatives))


def _two_taste_clusters() -> pd.DataFrame:
    # Users 0-7 like items 0-4, users 8-15 like items 5-9: a structure NeuMF must learn.
    rows = []
    for user in range(16):
        liked = range(0, 5) if user < 8 else range(5, 10)
        rows += [{"user_id": f"u{user}", "entity_id": f"i{item}", "event_type": "like"} for item in liked]
    return pd.DataFrame(rows)


def test_training_reduces_loss_and_learns_the_clusters():
    frame = _two_taste_clusters()
    # Each user holds out a different item of its cluster, so every item stays in training.
    user_num = frame["user_id"].str[1:].astype(int)
    item_num = frame["entity_id"].str[1:].astype(int)
    held_out = frame[item_num == (user_num % 5) + (user_num >= 8) * 5]
    train = frame.drop(held_out.index)

    model, user_to_idx, item_to_idx, _, losses = train_neumf(train, _tiny_config(), device=torch.device("cpu"), log=lambda _: None)
    assert losses[-1] < losses[0]

    seen = {
        user_to_idx[u]: {item_to_idx[i] for i in group["entity_id"]}
        for u, group in train.groupby("user_id")
    }
    users = sorted(seen)
    top = rank_top_k(model, users, [seen[u] for u in users], k=1)
    hits = sum(
        item_to_idx[row.entity_id] == top[users.index(user_to_idx[row.user_id])][0]
        for row in held_out.itertuples()
    )
    assert hits >= 12  # out of 16; random guessing among the 6 unseen items would get ~3


def test_rank_top_k_masks_training_items():
    torch.manual_seed(0)
    model = NeuMF(num_users=2, num_items=4, config=_tiny_config()).eval()
    top = rank_top_k(model, [0, 1], [{0, 1, 2}, set()], k=4)
    assert top[0][0] == 3  # the only unseen item comes first
    assert sorted(top[1]) == [0, 1, 2, 3]

"""Baselines and ablation for the hybrid ranker on an Amazon Reviews 2023 category.

Every signal of the hybrid score is run alone, the full hybrid is run with the
hand-set weights from the paper, and all 16 on/off combinations of the four
signals (semantic, EMA, LightGCN, KG) are run, each with its own weights
selected on a validation split. The empty combination is popularity alone and
the four-signal one is the tuned hybrid; removing one signal from the tuned
hybrid with the weights left unchanged is kept as a fixed-weight variant. All
variants share one chronological split, one candidate set (the whole training
catalogue) and one metric code path, so every number is directly comparable.

Amazon has no typed queries or declared profiles, so the online signals are
replaced by their closest offline counterparts:

  sem    the query is the user's most recent history item, i.e. the
         /recommend/item "more like this" path: cos(e_i, e_last)
  ema    the backend EMA update (alpha = 0.25, the first item initialises the
         profile) replayed over the history in time order: cos(e_i, u_T)
  graph  LightGCN trained on the training split only: cos(e_u, e_i)
  kg     entity overlap between the user's history and the candidate,
         min(1, |E(u) & E(i)| / max(3, |E(i)|)); entities are category-path
         nodes, store/brand and main category
  pop    log(1 + training count); reported as the reference baseline only

The declared-profile score has no Amazon counterpart and is not evaluated here.

Usage (Kaggle, after models.graph.datasets --out and
scripts.fetch_amazon_metadata --asins-from):

    python -m models.graph.evaluate_hybrid_ablation \\
        --dataset /kaggle/working/prepared_movies_and_tv_50000.parquet \\
        --metadata datasets/amazon/meta/Movies_and_TV.csv \\
        --report /kaggle/working/reports/movies_and_tv_ablation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.graph.evaluate_lightgcn import train_for_evaluation
from models.graph.lightgcn import LightGCNConfig

SIGNALS = ("sem", "ema", "graph", "kg", "pop")
# The four signals of the hybrid score, in the order the paper names them; the 16
# on/off combinations of these are the full-factorial ablation.
CORE_SIGNALS = ("sem", "ema", "graph", "kg")
SIGNAL_LETTERS = {"sem": "S", "ema": "E", "graph": "G", "kg": "K"}
# Eq. (hybrid) as deployed: backend.settings defaults.
PAPER_WEIGHTS = {"sem": 1.0, "graph": 0.20, "ema": 0.15, "kg": 0.08}
EMA_ALPHA = 0.25  # backend.settings.ema_alpha
# Per-combination tuning: the first signal of a combination has weight 1 (scores
# are compared by rank, so only the ratios matter); every other member takes a
# positive weight from this grid, so a combination never silently drops a signal.
RATIO_GRID = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
FULL_SUBSET = "S+E+G+K"
COMBINATION_PREFIX = "Combination "
# Breaks exact score ties (frequent for KG-only) the same way in every variant.
TIE_BREAK = 1e-6
MAX_ENTITY_SHARE = 0.5  # an entity on more than half the items carries no signal
KS = (10, 20)
METRICS = ("recall", "ndcg", "map", "mrr", "hit_rate", "precision")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_train_val_test(
    frame: pd.DataFrame,
    test_per_user: int = 2,
    val_per_user: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological leave-last-out: last n -> test, the one before -> validation.

    Users with fewer than test + val + 2 positives keep everything in train, so
    every evaluated user has at least two training items to build a history from.
    """
    frame = frame.sort_values(["user_id", "timestamp", "entity_id"], kind="stable")
    frame = frame.drop_duplicates(subset=["user_id", "entity_id"], keep="last")
    grouped = frame.groupby("user_id", sort=False)
    rank_from_end = grouped.cumcount(ascending=False)
    sizes = grouped["entity_id"].transform("size")
    eligible = sizes >= test_per_user + val_per_user + 2
    is_test = eligible & (rank_from_end < test_per_user)
    is_val = eligible & ~is_test & (rank_from_end < test_per_user + val_per_user)
    train = frame[~(is_test | is_val)].reset_index(drop=True)
    return train, frame[is_val].reset_index(drop=True), frame[is_test].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Item side: text embeddings and KG entities
# ---------------------------------------------------------------------------

def load_item_metadata(path: Path, item_ids: list[str]) -> pd.DataFrame:
    meta = pd.read_csv(path, dtype=str, keep_default_na=False)
    meta = meta.drop_duplicates(subset=["parent_asin"]).set_index("parent_asin")
    return meta.reindex(item_ids).fillna("")


def item_texts(meta: pd.DataFrame) -> list[str]:
    """Title, categories, description, creator (store): the catalogue's embedding_text order."""
    texts = []
    for row in meta.itertuples(index=False):
        parts = [row.title, row.categories, row.description[:1000], row.store]
        texts.append(". ".join(p.strip() for p in parts if p and p.strip()))
    return texts


def embed_items(texts: list[str], model_name: str, device: torch.device) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=str(device))
    vectors = model.encode(
        texts,
        batch_size=256,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    # An item without metadata has no semantic position; zero keeps cos = 0.
    vectors[[not t for t in texts]] = 0.0
    return vectors


def item_entities(meta: pd.DataFrame) -> list[set[str]]:
    entities = []
    for row in meta.itertuples(index=False):
        found = {
            "cat:" + c.strip().lower()
            for c in row.categories.split(", ")
            if c.strip()
        }
        if row.store.strip():
            found.add("store:" + row.store.strip().lower())
        if row.main_category.strip():
            found.add("main:" + row.main_category.strip().lower())
        entities.append(found)
    return entities


def entity_matrix(
    entities: list[set[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Sparse item x entity incidence (and its transpose) over informative entities."""
    counts: dict[str, int] = {}
    for found in entities:
        for entity in found:
            counts[entity] = counts.get(entity, 0) + 1
    ceiling = MAX_ENTITY_SHARE * len(entities)
    vocab = {e: i for i, e in enumerate(sorted(e for e, c in counts.items() if 2 <= c <= ceiling))}

    rows, cols = [], []
    for item_idx, found in enumerate(entities):
        for entity in found:
            col = vocab.get(entity)
            if col is not None:
                rows.append(item_idx)
                cols.append(col)
    num_items, num_entities = len(entities), max(1, len(vocab))
    index = torch.as_tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    item_by_entity = torch.sparse_coo_tensor(index, values, (num_items, num_entities)).coalesce()
    entity_by_item = torch.sparse_coo_tensor(index.flip(0), values, (num_entities, num_items)).coalesce()
    per_item = torch.bincount(index[0], minlength=num_items).float()
    return item_by_entity, entity_by_item, per_item, len(vocab)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Context:
    device: torch.device
    item_text: torch.Tensor  # N x 384, rows zero when no metadata
    has_text: torch.Tensor  # N bool
    user_graph: torch.Tensor  # U x d, row-normalised LightGCN
    item_graph: torch.Tensor  # N x d
    item_by_entity: torch.Tensor  # sparse N x E
    entity_by_item: torch.Tensor  # sparse E x N
    entities_per_item: torch.Tensor  # N
    popularity: torch.Tensor  # N, log(1 + train count)

    @property
    def num_items(self) -> int:
        return int(self.item_text.shape[0])


def batch_signals(
    user_rows: list[int],
    histories: list[np.ndarray],
    ctx: Context,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """All signal matrices (B x N) for one batch of users, plus the seen-item mask."""
    device, num_items = ctx.device, ctx.num_items
    batch = len(user_rows)
    max_len = max(len(h) for h in histories)
    padded = torch.full((batch, max_len), -1, dtype=torch.long)
    for row, history in enumerate(histories):
        padded[row, : len(history)] = torch.as_tensor(history, dtype=torch.long)
    padded = padded.to(device)
    valid = padded >= 0
    safe = padded.clamp(min=0)

    seen = torch.zeros(batch, num_items, dtype=torch.bool, device=device)
    row_ids = torch.arange(batch, device=device).unsqueeze(1).expand_as(padded)
    seen[row_ids[valid], padded[valid]] = True

    # sem: the latest history item that has text plays the query.
    with_text = valid & ctx.has_text[safe]
    last_pos = max_len - 1 - torch.flip(with_text.int(), dims=[1]).argmax(dim=1)
    query = ctx.item_text[safe[torch.arange(batch, device=device), last_pos]]
    query = query * with_text.any(dim=1, keepdim=True)
    sem = query @ ctx.item_text.T

    # ema: backend.ema_recommender.update_profile_vector replayed in time order.
    profile = torch.zeros(batch, ctx.item_text.shape[1], device=device)
    started = torch.zeros(batch, dtype=torch.bool, device=device)
    for step in range(max_len):
        active = with_text[:, step]
        if not active.any():
            continue
        item_vec = ctx.item_text[safe[:, step]]
        blended = torch.where(
            started.unsqueeze(1),
            (1.0 - EMA_ALPHA) * profile + EMA_ALPHA * item_vec,
            item_vec,
        )
        profile = torch.where(active.unsqueeze(1), F.normalize(blended, dim=1), profile)
        started |= active
    ema = profile @ ctx.item_text.T

    graph = ctx.user_graph[torch.as_tensor(user_rows, device=device)] @ ctx.item_graph.T

    # kg: entities the user has touched, overlapped with each candidate's entities.
    touched = torch.sparse.mm(ctx.entity_by_item, seen.T.float()) > 0  # E x B
    overlap = torch.sparse.mm(ctx.item_by_entity, touched.float()).T  # B x N
    kg = (overlap / ctx.entities_per_item.clamp(min=3.0)).clamp(max=1.0)

    pop = ctx.popularity.unsqueeze(0).expand(batch, -1)
    return {"sem": sem, "ema": ema, "graph": graph, "kg": kg, "pop": pop}, seen


def combine(signals: dict[str, torch.Tensor], weights: dict[str, float], seen: torch.Tensor) -> torch.Tensor:
    scores = TIE_BREAK * signals["pop"]
    for name, weight in weights.items():
        if weight:
            scores = scores + weight * signals[name]
    return scores.masked_fill(seen, float("-inf"))


def per_user_metrics(scores: torch.Tensor, relevant: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-user metrics at every K in KS; ``relevant`` is the B x N held-out mask."""
    top = torch.topk(scores, max(KS), dim=1).indices
    hits = relevant.gather(1, top).float()
    num_relevant = relevant.sum(dim=1).float()
    ranks = torch.arange(1, max(KS) + 1, device=scores.device, dtype=torch.float32)
    discounts = 1.0 / torch.log2(ranks + 1.0)

    out: dict[str, torch.Tensor] = {}
    for k in KS:
        h = hits[:, :k]
        hit_count = h.sum(dim=1)
        ideal = torch.minimum(num_relevant, torch.tensor(float(k), device=scores.device))
        ideal_dcg = torch.cumsum(discounts[:k], dim=0)[(ideal - 1).long().clamp(min=0)]
        first = h.argmax(dim=1).float()
        precision_at = torch.cumsum(h, dim=1) / ranks[:k]
        out[f"recall@{k}"] = hit_count / num_relevant
        out[f"precision@{k}"] = hit_count / k
        out[f"hit_rate@{k}"] = (hit_count > 0).float()
        out[f"ndcg@{k}"] = (h * discounts[:k]).sum(dim=1) / ideal_dcg
        out[f"mrr@{k}"] = torch.where(hit_count > 0, 1.0 / (first + 1.0), torch.zeros_like(first))
        out[f"map@{k}"] = (precision_at * h).sum(dim=1) / ideal
    return out


# ---------------------------------------------------------------------------
# Evaluation loops
# ---------------------------------------------------------------------------

def _batches(users: list[str], size: int):
    for start in range(0, len(users), size):
        yield users[start : start + size]


def _relevant_mask(batch: list[str], relevant: dict[str, np.ndarray], num_items: int, device) -> torch.Tensor:
    mask = torch.zeros(len(batch), num_items, dtype=torch.bool, device=device)
    for row, user in enumerate(batch):
        mask[row, torch.as_tensor(relevant[user], dtype=torch.long, device=device)] = True
    return mask


def signal_subsets() -> list[tuple[str, ...]]:
    """The 15 non-empty combinations of the four signals, smallest first.

    The 16th combination, the empty one, ranks by popularity alone (the tie-break
    every variant shares), so it is the "Most popular" baseline.
    """
    return [
        subset
        for size in range(1, len(CORE_SIGNALS) + 1)
        for subset in itertools.combinations(CORE_SIGNALS, size)
    ]


def subset_label(subset: tuple[str, ...]) -> str:
    return "+".join(SIGNAL_LETTERS[s] for s in subset) if subset else "none"


def tune_subsets(
    users: list[str],
    user_rows: dict[str, int],
    histories: dict[str, np.ndarray],
    relevant: dict[str, np.ndarray],
    ctx: Context,
    batch_size: int,
    select_by: str = "ndcg@20",
) -> tuple[dict[str, dict[str, float]], dict[str, list[dict]]]:
    """Validation-selected weights for every combination, in one pass over the users.

    All combinations share the signal matrices of a batch, so tuning the 15 of them
    costs less than the single four-signal grid search.
    """
    combos: list[tuple[str, dict[str, float]]] = []
    for subset in signal_subsets():
        first, rest = subset[0], subset[1:]
        for ratios in itertools.product(RATIO_GRID, repeat=len(rest)):
            combos.append((subset_label(subset), {first: 1.0, **dict(zip(rest, ratios))}))
    totals = np.zeros((len(combos), 2))
    for batch in _batches(users, batch_size):
        signals, seen = batch_signals([user_rows[u] for u in batch], [histories[u] for u in batch], ctx)
        mask = _relevant_mask(batch, relevant, ctx.num_items, ctx.device)
        for idx, (_, weights) in enumerate(combos):
            metrics = per_user_metrics(combine(signals, weights, seen), mask)
            totals[idx, 0] += float(metrics[select_by].sum())
            totals[idx, 1] += float(metrics["recall@20"].sum())
    totals /= len(users)

    chosen: dict[str, dict[str, float]] = {}
    boards: dict[str, list[dict]] = {}
    for label in dict.fromkeys(label for label, _ in combos):
        members = [i for i, (name, _) in enumerate(combos) if name == label]
        order = sorted(members, key=lambda i: -totals[i, 0])  # stable: first grid entry wins ties
        chosen[label] = combos[order[0]][1]
        boards[label] = [
            {"weights": combos[i][1], select_by: float(totals[i, 0]), "recall@20": float(totals[i, 1])}
            for i in order[:5]
        ]
    return chosen, boards


def evaluate_configs(
    configs: dict[str, dict[str, float]],
    users: list[str],
    user_rows: dict[str, int],
    histories: dict[str, np.ndarray],
    relevant: dict[str, np.ndarray],
    ctx: Context,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    collected: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in configs}
    for batch in _batches(users, batch_size):
        signals, seen = batch_signals([user_rows[u] for u in batch], [histories[u] for u in batch], ctx)
        mask = _relevant_mask(batch, relevant, ctx.num_items, ctx.device)
        for name, weights in configs.items():
            for metric, values in per_user_metrics(combine(signals, weights, seen), mask).items():
                collected[name].setdefault(metric, []).append(values.cpu().numpy())
    return {
        name: {metric: np.concatenate(parts) for metric, parts in metrics.items()}
        for name, metrics in collected.items()
    }


def paired_tests(per_user: dict[str, dict[str, np.ndarray]], reference: str, metric: str) -> dict[str, float]:
    from scipy.stats import wilcoxon

    tests = {}
    base = per_user[reference][metric]
    for name, metrics in per_user.items():
        if name == reference:
            continue
        diff = base - metrics[metric]
        tests[name] = 1.0 if not np.any(diff) else float(wilcoxon(base, metrics[metric]).pvalue)
    return tests


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _index_rows(rows: pd.DataFrame, item_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    mapped = rows.assign(item_idx=rows["entity_id"].astype(str).map(item_to_idx)).dropna(subset=["item_idx"])
    mapped = mapped.sort_values(["user_id", "timestamp"], kind="stable")
    return {
        user: group["item_idx"].to_numpy(dtype=np.int64)
        for user, group in mapped.groupby("user_id", sort=False)
    }


def _graph_embeddings(train: pd.DataFrame, config: LightGCNConfig, cache: Path | None):
    if cache and cache.exists():
        stored = np.load(cache, allow_pickle=True)
        print(f"reusing LightGCN embeddings from {cache}", flush=True)
        return stored["users"], stored["items"], list(stored["user_ids"]), list(stored["item_ids"]), None
    started = time.perf_counter()
    users, items, user_ids, item_ids, _ = train_for_evaluation(train, config)
    seconds = time.perf_counter() - started
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, users=users, items=items, user_ids=np.array(user_ids), item_ids=np.array(item_ids))
    return users, items, user_ids, item_ids, seconds


def run(args: argparse.Namespace) -> dict:
    timings: dict[str, float] = {}
    clock = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from models.graph.datasets import prepare_interactions

    # A prepared cache (models.graph.datasets --out) is reused as is; a raw
    # Amazon file goes through the same binarise + k-core as the benchmark.
    interactions = prepare_interactions(
        args.dataset, user_core=args.user_core, item_core=args.item_core,
        max_users=args.max_users, seed=args.seed,
    )

    train, val, test = split_train_val_test(interactions, args.test_per_user, args.val_per_user)
    print(f"split: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    config = LightGCNConfig(
        embedding_dim=args.embedding_dim, num_layers=args.layers, learning_rate=args.lr,
        weight_decay=args.weight_decay, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
    )
    user_graph, item_graph, user_ids, item_ids, train_seconds = _graph_embeddings(train, config, args.graph_cache)
    timings["lightgcn_train_s"] = train_seconds
    user_rows = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}

    stage = time.perf_counter()
    meta = load_item_metadata(args.metadata, item_ids)
    texts = item_texts(meta)
    item_text = embed_items(texts, args.encoder, device)
    timings["embed_items_s"] = time.perf_counter() - stage
    item_by_entity, entity_by_item, per_item, num_entities = entity_matrix(item_entities(meta), device)

    counts = train["entity_id"].astype(str).map(item_to_idx).dropna().astype(int).value_counts()
    popularity = np.zeros(len(item_ids), dtype=np.float32)
    popularity[counts.index.to_numpy()] = np.log1p(counts.to_numpy())

    ctx = Context(
        device=device,
        item_text=torch.as_tensor(item_text, device=device),
        has_text=torch.as_tensor([bool(t) for t in texts], device=device),
        user_graph=torch.as_tensor(user_graph, dtype=torch.float32, device=device),
        item_graph=torch.as_tensor(item_graph, dtype=torch.float32, device=device),
        item_by_entity=item_by_entity,
        entity_by_item=entity_by_item,
        entities_per_item=per_item,
        popularity=torch.as_tensor(popularity, device=device),
    )

    train_hist = _index_rows(train, item_to_idx)
    val_rel = _index_rows(val, item_to_idx)
    test_rel = _index_rows(test, item_to_idx)
    test_hist = _index_rows(pd.concat([train, val]), item_to_idx)

    rng = np.random.default_rng(args.seed)
    val_users = sorted(u for u in val_rel if u in user_rows and u in train_hist)
    if args.val_users and len(val_users) > args.val_users:
        val_users = sorted(rng.choice(val_users, size=args.val_users, replace=False).tolist())
    test_users = sorted(u for u in test_rel if u in user_rows and u in test_hist)
    print(f"users: validation={len(val_users):,} test={len(test_users):,}", flush=True)

    # Every combination gets its own validation-selected weights; the four-signal
    # one is the tuned hybrid, so the headline row and the full-factorial row are
    # the same run.
    stage = time.perf_counter()
    subset_weights, subset_boards = tune_subsets(val_users, user_rows, train_hist, val_rel, ctx, args.eval_batch)
    timings["weight_tuning_s"] = time.perf_counter() - stage
    tuned = subset_weights[FULL_SUBSET]
    print(f"tuned weights: {tuned}", flush=True)
    print(f"combination weights: {subset_weights}", flush=True)

    configs: dict[str, dict[str, float]] = {
        "Popularity": {"pop": 1.0},
        "Semantic only": {"sem": 1.0},
        "EMA only": {"ema": 1.0},
        "LightGCN only": {"graph": 1.0},
        "KG only": {"kg": 1.0},
        "Hybrid (paper weights)": dict(PAPER_WEIGHTS),
        "Hybrid (tuned)": dict(tuned),
    }
    for signal in ("sem", "ema", "graph", "kg"):
        configs[f"Hybrid (tuned) w/o {signal}"] = {**tuned, signal: 0.0}
    # Full-factorial ablation: all 16 on/off combinations of the four signals, each
    # with its own validation-selected weights (the empty one is popularity alone).
    configs[COMBINATION_PREFIX + "none"] = {}
    for label, weights in subset_weights.items():
        configs[COMBINATION_PREFIX + label] = weights

    stage = time.perf_counter()
    per_user = evaluate_configs(configs, test_users, user_rows, test_hist, test_rel, ctx, args.eval_batch)
    timings["test_eval_s"] = time.perf_counter() - stage
    timings["rerank_ms_per_user"] = 1000.0 * timings["test_eval_s"] / max(1, len(test_users) * len(configs))
    timings["total_s"] = time.perf_counter() - clock

    all_results = {
        name: {metric: float(values.mean()) for metric, values in metrics.items()}
        for name, metrics in per_user.items()
    }
    results = {n: m for n, m in all_results.items() if not n.startswith(COMBINATION_PREFIX)}
    factorial_users = {n: m for n, m in per_user.items() if n.startswith(COMBINATION_PREFIX)}
    factorial_tests = paired_tests(factorial_users, COMBINATION_PREFIX + FULL_SUBSET, "ndcg@20")
    factorial = {
        "full": FULL_SUBSET,
        "order": ["none"] + [subset_label(s) for s in signal_subsets()],
        "weights": {"none": {}, **subset_weights},
        "results": {
            n[len(COMBINATION_PREFIX):]: m for n, m in all_results.items() if n.startswith(COMBINATION_PREFIX)
        },
        "wilcoxon_vs_full_ndcg@20": {n[len(COMBINATION_PREFIX):]: p for n, p in factorial_tests.items()},
        "validation_leaderboard": subset_boards,
        "grid": {"first signal": 1.0, "others": RATIO_GRID},
    }
    report = {
        "dataset": args.dataset.name,
        "device": str(device),
        "graph": {
            "users": len(user_ids), "items": len(item_ids),
            "train": len(train), "val": len(val), "test": len(test),
            "items_with_text": int(sum(bool(t) for t in texts)),
            "kg_entities": num_entities,
        },
        "protocol": {
            "split": f"chronological; last {args.test_per_user} positives -> test, "
                     f"previous {args.val_per_user} -> validation, rest -> train",
            "candidates": "full training catalogue minus the user's history",
            "val_users": len(val_users), "test_users": len(test_users),
            "grid": {"first signal of a combination": 1.0, "other signals": RATIO_GRID},
            "selected_by": "validation NDCG@20",
            "ema_alpha": EMA_ALPHA, "encoder": args.encoder,
        },
        "lightgcn": config.__dict__,
        "tuned_weights": tuned,
        "validation_leaderboard": subset_boards[FULL_SUBSET],
        "results": results,
        "wilcoxon_vs_tuned_ndcg@20": paired_tests(
            {n: m for n, m in per_user.items() if not n.startswith(COMBINATION_PREFIX)}, "Hybrid (tuned)", "ndcg@20"
        ),
        "factorial": factorial,
        "timings": timings,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        args.report.with_suffix(".md").write_text(markdown_table(report), encoding="utf-8")
        print(f"report -> {args.report}")
    print(markdown_table(report))
    print(json.dumps(report), flush=True)
    return report


def markdown_table(report: dict) -> str:
    columns = [f"{m}@{k}" for k in KS for m in ("recall", "ndcg", "map", "mrr")]
    pvalues = report["wilcoxon_vs_tuned_ndcg@20"]
    lines = [
        f"### {report['dataset']}  (tuned weights: {report['tuned_weights']})",
        "",
        "| Method | " + " | ".join(c.replace("_", " ").upper().replace("@", "@") for c in columns) + " | p (NDCG@20) |",
        "|---" * (len(columns) + 2) + "|",
    ]
    for name, metrics in report["results"].items():
        p = pvalues.get(name)
        cells = " | ".join(f"{metrics[c]:.4f}" for c in columns)
        lines.append(f"| {name} | {cells} | {'--' if p is None else f'{p:.1e}'} |")
    factorial = report.get("factorial")
    if factorial:
        fp = factorial["wilcoxon_vs_full_ndcg@20"]
        lines += [
            "",
            "#### All 16 signal combinations (S semantic, E EMA, G LightGCN, K KG; each tuned on validation)",
            "",
            "| Combination | weights | " + " | ".join(c.upper() for c in columns) + " | p vs full (NDCG@20) |",
            "|---" * (len(columns) + 3) + "|",
        ]
        for label in factorial["order"]:
            metrics = factorial["results"][label]
            weights = ", ".join(f"{k}={v:g}" for k, v in factorial["weights"][label].items()) or "popularity"
            p = fp.get(label)
            cells = " | ".join(f"{metrics[c]:.4f}" for c in columns)
            lines.append(f"| {label} | {weights} | {cells} | {'--' if p is None else f'{p:.1e}'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--graph-cache", type=Path, default=None,
                        help="Save/reuse the LightGCN embeddings (.npz) across reruns.")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--test-per-user", type=int, default=2)
    parser.add_argument("--val-per-user", type=int, default=1)
    parser.add_argument("--val-users", type=int, default=10000, help="0 = all")
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--user-core", type=int, default=10)
    parser.add_argument("--item-core", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

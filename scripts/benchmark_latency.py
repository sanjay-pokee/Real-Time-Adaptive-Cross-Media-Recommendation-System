"""Measure end-to-end and per-stage latency of POST /recommend, plus throughput.

Requests go through FastAPI's TestClient, so validation, the full recommender
and response serialisation are all inside the timing; only the network hop is
not. Each recommender stage is wrapped with a timer on the live app instance,
which gives the per-stage breakdown without touching backend code.

The personalised path depends on MySQL, and every MySQL call in the
recommender swallows its own errors and falls back to the unpersonalised
ranking. A run with MySQL down would therefore look fast for the wrong reason,
so the script refuses to start unless MySQL answers and a demo user actually
gets EMA and graph scores back.

Usage (stop the dev server first - embedded Qdrant allows one process):
    python -m scripts.benchmark_latency
    python -m scripts.benchmark_latency --requests 1000 --concurrency 1 4 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "reports" / "latency_benchmark.json"

QUERIES = [
    "James Cameron",
    "Taylor Swift",
    "business analytics and learning",
    "fun animation adventure",
    "space adventure with aliens",
    "Christopher Nolan",
    "J.K. Rowling",
    "romantic comedy",
    "dark psychological thriller",
    "history of the roman empire",
    "relaxing acoustic music",
    "self-help and productivity",
    "superhero action",
    "learning python programming",
    "classic rock",
    "fantasy dragons and magic",
]
USERS = [
    None,  # anonymous: semantic + creator + lexical + KG only
    "user_scifi",
    "user_books_learning",
    "user_family",
    "user_music_pop",
    "user_fantasy",
    "user_dark_thriller",
]
STAGES = {
    "_search_person_matches": "creator match",
    "_search_lexical_matches": "lexical match",
    "_encode": "query encoding",
    "_search_vector": "Qdrant ANN search",
    "_personalize_results": "MySQL preference rerank",
    "_graph_rerank": "LightGCN rerank",
    "_ema_rerank": "EMA rerank",
    "_knowledge_graph_rerank": "KG/profile rerank",
}


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _instrument(recommender, timings: dict[str, list[float]]) -> None:
    """Wrap each pipeline stage on the live instance with a perf_counter timer."""

    def wrap(name, fn):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings[name].append(1000.0 * (time.perf_counter() - started))

        return timed

    for name in STAGES:
        if name == "_encode":
            continue
        setattr(recommender, name, wrap(name, getattr(recommender, name)))

    model = recommender.model
    original_encode = model.encode
    model.encode = wrap("_encode", original_encode)


def _check_personalisation(client) -> None:
    from backend.mysql_store import MySQLStore

    try:
        vector = MySQLStore().get_user_ema_vector("user_scifi")
    except Exception as exc:  # noqa: BLE001 - report any connection failure
        raise SystemExit(
            f"MySQL is not reachable ({exc.__class__.__name__}: {exc}).\n"
            "Start the MySQL80 service; otherwise every personalised request silently "
            "falls back to the unpersonalised path and the latency is meaningless."
        )
    if not vector:
        raise SystemExit("user_scifi has no EMA vector. Run scripts.seed_demo_interactions first.")

    body = client.post("/recommend", json={"query": "space adventure", "top_k": 10, "user_id": "user_scifi"}).json()
    results = body.get("results", [])
    if not results or results[0].get("ema_score") is None:
        raise SystemExit("Personalised request returned no ema_score; the EMA path is not active.")
    if all(r.get("graph_score") is None for r in results):
        print("WARNING: no graph_score in results - LightGCN artifact missing or user not in graph.")


def run(args: argparse.Namespace) -> dict:
    os.chdir(PROJECT_ROOT)
    from fastapi.testclient import TestClient

    from backend.app import app

    report: dict = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "settings": {"top_k": args.top_k, "requests": args.requests, "warmup": args.warmup},
    }
    try:
        import torch

        report["machine"]["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pass

    with TestClient(app) as client:
        recommender = app.state.recommender
        report["catalog_items"] = int(len(recommender.catalog))
        _check_personalisation(client)

        workload = list(itertools.islice(itertools.cycle(itertools.product(QUERIES, USERS)), args.warmup + args.requests))

        def call(query_user):
            query, user = query_user
            payload = {"query": query, "top_k": args.top_k}
            if user:
                payload["user_id"] = user
            started = time.perf_counter()
            response = client.post("/recommend", json=payload)
            elapsed = 1000.0 * (time.perf_counter() - started)
            if response.status_code != 200:
                raise RuntimeError(f"{response.status_code}: {response.text[:200]}")
            return user is not None, elapsed

        for item in workload[: args.warmup]:
            call(item)

        stage_times: dict[str, list[float]] = defaultdict(list)
        _instrument(recommender, stage_times)

        anonymous, personalised = [], []
        for item in workload[args.warmup :]:
            is_personal, elapsed = call(item)
            (personalised if is_personal else anonymous).append(elapsed)

        report["end_to_end"] = {
            "all": _percentiles(anonymous + personalised),
            "anonymous": _percentiles(anonymous),
            "personalised": _percentiles(personalised),
        }
        report["stages"] = {
            STAGES[name]: _percentiles(values) for name, values in stage_times.items() if values
        }

        throughput = {}
        for workers in args.concurrency:
            batch = workload[args.warmup : args.warmup + args.throughput_requests]
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(call, batch))
            seconds = time.perf_counter() - started
            throughput[str(workers)] = {"requests": len(batch), "seconds": seconds, "req_per_s": len(batch) / seconds}
        report["throughput"] = throughput

        # EMA update cost: the per-event work behind POST /interactions, without the DB write.
        ema_store = recommender.ema_store
        if ema_store is not None:
            ids = list(ema_store.item_to_row)[:2000]
            vector = None
            updates = []
            for entity_id in ids:
                started = time.perf_counter()
                vector = ema_store.update_profile_vector(vector, entity_id, "like", None)
                updates.append(1000.0 * (time.perf_counter() - started))
            report["ema_update"] = _percentiles(updates)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_PATH.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"report -> {REPORT_PATH}")
    return report


def markdown(report: dict) -> str:
    lines = [
        f"Machine: {report['machine'].get('processor') or report['machine']['platform']}, "
        f"{report['machine']['cpu_count']} logical CPUs, encoder on {report['machine'].get('torch_device', '?')}; "
        f"catalog {report['catalog_items']:,} items; top_k={report['settings']['top_k']}",
        "",
        "| Path / stage | n | mean ms | p50 ms | p95 ms | p99 ms |",
        "|---|---|---|---|---|---|",
    ]
    rows = [(f"POST /recommend ({k})", v) for k, v in report["end_to_end"].items()]
    rows += [(f"  stage: {k}", v) for k, v in report["stages"].items()]
    if "ema_update" in report:
        rows.append(("EMA profile update (per event)", report["ema_update"]))
    for name, stats in rows:
        lines.append(
            f"| {name} | {stats['n']} | {stats['mean_ms']:.2f} | {stats['p50_ms']:.2f} | "
            f"{stats['p95_ms']:.2f} | {stats['p99_ms']:.2f} |"
        )
    lines += ["", "| Concurrent workers | requests | req/s |", "|---|---|---|"]
    for workers, stats in report["throughput"].items():
        lines.append(f"| {workers} | {stats['requests']} | {stats['req_per_s']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--requests", type=int, default=700)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--throughput-requests", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

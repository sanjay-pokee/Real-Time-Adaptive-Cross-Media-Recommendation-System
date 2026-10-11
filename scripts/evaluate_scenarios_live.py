"""Run the paper's prototype scenarios through the deployed POST /recommend.

``scripts.evaluate_recommendation_table`` scores a candidate list it builds
itself by substring-matching the catalogue against each scenario's expected
terms, then reranks it with the KG scorer; Qdrant, Sentence-BERT, LightGCN,
EMA and the MySQL profiles never take part. Its numbers therefore describe the
KG scorer on a hand-picked pool, not the system. This script sends the same
scenarios to the real API, with the scenario's demo user, and measures what
that user would actually be shown.

Metrics per scenario (top_k = 10):
  person_hits@5     top-5 results whose creators contain the queried person
                    (person queries only)
  profile_aligned@5 top-5 results with profile_score > 0
  avg_kg@5          mean kg_score of the top 5
  avg_profile@5     mean profile_score of the top 5
  domains@10        content types present in the top 10
  personalised@10   top-10 results carrying an EMA score / a LightGCN score

Each request declares what the frontend would: the viewer age implied by the
demo profile's age group (an undeclared viewer is capped at the teen ceiling,
which hides the adult-default health, industry and finance catalogues) and,
for the three vertical scenarios, the domain - as the frontend's query chips
do with ``domain: 'health', age: 30``.

Needs MySQL (demo profiles and interactions) and the same Qdrant the backend
uses; with an embedded store, stop the dev server first.

Usage:
    python -m scripts.evaluate_scenarios_live
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

from scripts.evaluate_recommendation_table import SCENARIOS
from scripts.seed_demo_interactions import USER_PROFILES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
JSON_PATH = REPORT_DIR / "scenario_evaluation_live.json"
MD_PATH = REPORT_DIR / "scenario_evaluation_live.md"
PERSON_QUERIES = {"James Cameron", "Taylor Swift"}
TOP_K = 10
# Declared viewer age per profile age group; 8 matches the frontend's "Kids" chip.
AGE_BY_GROUP = {"young_adult": 20, "adult": 30, "family": 8, "teen": 15}
# Vertical scenarios declare their domain, as the frontend's query chips do.
DOMAIN_BY_SCENARIO = {
    "Health product discovery": "health",
    "Industrial supply lookup": "industry",
    "Finance literacy discovery": "finance",
}


def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).casefold().split())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(client) -> list[dict]:
    from backend.mysql_store import MySQLStore

    store = MySQLStore()
    rows = []
    for scenario in SCENARIOS:
        user_id = scenario["user_id"]
        if not store.get_user_profile_preferences(user_id):
            raise SystemExit(f"{user_id} has no profile in MySQL; run scripts.seed_demo_interactions first.")

        age = AGE_BY_GROUP[USER_PROFILES[user_id]["age_group"]]
        domain = DOMAIN_BY_SCENARIO.get(scenario["name"])
        payload = {"query": scenario["query"], "top_k": TOP_K, "user_id": user_id, "age": age}
        if domain:
            payload["domain"] = domain
        response = client.post("/recommend", json=payload)
        if response.status_code != 200:
            raise SystemExit(f"{scenario['name']}: HTTP {response.status_code} {response.text[:200]}")
        results = response.json()["results"]
        top5, top10 = results[:5], results[:TOP_K]

        person = scenario["query"] if scenario["query"] in PERSON_QUERIES else None
        rows.append({
            "scenario": scenario["name"],
            "query": scenario["query"],
            "user_id": user_id,
            "age": age,
            "domain": domain,
            "person_hits@5": (
                sum(1 for r in top5 if _norm(person) in _norm(r.get("creators", ""))) if person else None
            ),
            "profile_aligned@5": sum(1 for r in top5 if (r.get("profile_score") or 0.0) > 0),
            "avg_kg@5": round(_mean([r.get("kg_score") or 0.0 for r in top5]), 3),
            "avg_profile@5": round(_mean([r.get("profile_score") or 0.0 for r in top5]), 3),
            "domains@10": sorted({str(r.get("content_type", "")) for r in top10}),
            "ema_scored@10": sum(1 for r in top10 if r.get("ema_score") is not None),
            "graph_scored@10": sum(1 for r in top10 if r.get("graph_score") is not None),
            "top_result": top5[0]["title"] if top5 else "",
            "top10": [(r.get("content_type"), r.get("title")) for r in top10],
        })
    return rows


def markdown(rows: list[dict]) -> str:
    lines = [
        "| Scenario | Query | Age / domain | PersonHit@5 | ProfileAlign@5 | Avg KG@5 | Avg profile@5 | Domains@10 | EMA/graph-scored@10 | Top result |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        hits = "--" if r["person_hits@5"] is None else f"{r['person_hits@5']}/5"
        lines.append(
            f"| {r['scenario']} | {r['query']} | {r['age']} / {r['domain'] or '-'} | {hits} | {r['profile_aligned@5']}/5 | {r['avg_kg@5']:.3f} | "
            f"{r['avg_profile@5']:.3f} | {', '.join(r['domains@10'])} | {r['ema_scored@10']}/{r['graph_scored@10']} | "
            f"{r['top_result']} |"
        )
    multi = sum(1 for r in rows if len(r["domains@10"]) >= 2)
    person = [r["person_hits@5"] for r in rows if r["person_hits@5"] is not None]
    lines += [
        "",
        f"Cross-domain rate (top 10 spans >= 2 content types): {multi}/{len(rows)}",
        f"Mean PersonHit@5 over person queries: {_mean(person) / 5:.2f}",
        f"Mean ProfileAlign@5: {_mean([r['profile_aligned@5'] for r in rows]) / 5:.2f}",
        f"Mean KG@5: {_mean([r['avg_kg@5'] for r in rows]):.3f}   Mean profile@5: {_mean([r['avg_profile@5'] for r in rows]):.3f}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    os.chdir(PROJECT_ROOT)
    from fastapi.testclient import TestClient

    from backend.app import app

    with TestClient(app) as client:
        rows = evaluate(client)
    REPORT_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    MD_PATH.write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))
    for r in rows:
        print(f"\n{r['scenario']} ({r['query']}, {r['user_id']}):")
        for content_type, title in r["top10"]:
            print(f"   [{content_type}] {title}")
    print(f"\nreport -> {JSON_PATH}")


if __name__ == "__main__":
    main()

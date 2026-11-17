"""Live API walkthrough for the review: every core endpoint, real responses, one command.

    python -m scripts.api_demo            # run all steps
    python -m scripts.api_demo --pause    # wait for Enter between steps (for presenting)

Needs the API running (scripts/demo_up.ps1). Every request is a normal HTTP call
to the same endpoints the React app uses; nothing is mocked. The interaction
step likes an item and then un-likes it, so the demo profile ends unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"
USER = "user_scifi"


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | None, float]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, payload = resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as err:
        status, payload = err.code, json.loads(err.read() or b"null")
    return status, payload, (time.perf_counter() - started) * 1000


def heading(n: int, title: str, request: str) -> None:
    print(f"\n{'=' * 78}\n STEP {n}. {title}\n   {request}\n{'-' * 78}")


def show_results(payload: dict, limit: int = 3) -> None:
    if payload.get("corrected_query"):
        print(f"   corrected query: {payload['corrected_query']}")
    if payload.get("advisory"):
        print(f"   advisory: {payload['advisory']}")
    for r in payload.get("results", [])[:limit]:
        print(f"   - {r['title'][:46]:46} [{r.get('content_type')}, {r.get('maturity')}, {r.get('audience_min_age')}+]")
        signals = []
        for key, label in (("semantic_score", "semantic"), ("preference_bonus", "profile bonus"), ("graph_score", "lightgcn"),
                           ("ema_score", "ema"), ("kg_score", "kg"), ("profile_score", "profile")):
            if r.get(key) is not None:
                signals.append(f"{label} {r[key]:.3f}")
        print(f"       score {r['score']:.3f} = " + ", ".join(signals))
        if r.get("kg_explanation"):
            print(f"       why: {r['kg_explanation']}")


def pause(enabled: bool) -> None:
    if enabled:
        input("\n   [Enter] for the next step ")


def main() -> None:
    global BASE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    heading(1, "Liveness", "GET /health")
    status, payload, ms = call("GET", "/health")
    print(f"   {status} in {ms:.0f} ms -> {payload}")
    if status != 200:
        sys.exit("API is not up. Start it with .\\scripts\\demo_up.ps1")
    pause(args.pause)

    heading(2, "Domain packs (config-driven)", "GET /domains")
    status, payload, ms = call("GET", "/domains")
    for d in payload.get("domains", []):
        print(f"   {d['name']:14} types={','.join(d.get('content_types', []))}  risk_tier={d.get('risk_tier')}  advisory={d.get('advisory')}")
    pause(args.pause)

    heading(3, "Knowledge graph size", "GET /kg/stats")
    status, payload, ms = call("GET", "/kg/stats")
    print(f"   {status} in {ms:.0f} ms -> nodes={payload['nodes']:,}  edges={payload['edge_total']:,}")
    print(f"   entities: {payload['entities']}")
    pause(args.pause)

    body = {"query": "christopher nolan", "user_id": USER, "top_k": 3, "age": 30}
    heading(4, "Recommend: creator query, personalised", f"POST /recommend {json.dumps(body)}")
    status, payload, ms = call("POST", "/recommend", body)
    print(f"   {status} in {ms:.0f} ms")
    show_results(payload)
    pause(args.pause)

    body = {"query": "christopher nolan", "user_id": USER, "top_k": 3, "age": 30, "safe_mode": True}
    heading(5, "Eligibility: same query with safe mode (child ceiling 7+)", f"POST /recommend {json.dumps(body)}")
    status, payload, ms = call("POST", "/recommend", body)
    print(f"   {status} in {ms:.0f} ms  -> no 13+ Nolan film can appear")
    show_results(payload)
    pause(args.pause)

    body = {"query": "vitamin d supplement", "domain": "health", "top_k": 3, "age": 30}
    heading(6, "Regulated domain: advisory attached", f"POST /recommend {json.dumps(body)}")
    status, payload, ms = call("POST", "/recommend", body)
    print(f"   {status} in {ms:.0f} ms")
    show_results(payload)
    pause(args.pause)

    heading(7, "Validation errors (never an unfiltered fallback)", 'POST /recommend {"query": ""} and {"query": "x", "domain": "space"}')
    for bad in ({"query": "", "top_k": 3}, {"query": "x", "domain": "space", "top_k": 3}):
        status, payload, ms = call("POST", "/recommend", bad)
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        print(f"   {json.dumps(bad):45} -> {status}  {str(detail)[:70]}")
    pause(args.pause)

    status, payload, _ = call("POST", "/recommend", {"query": "inception", "top_k": 1, "age": 30, "content_type": "movie"})
    item = payload["results"][0]
    gid = item["global_id"]
    state_path = f"/interactions/{USER}/{urllib.parse.quote(gid, safe=':')}"
    like = {"user_id": USER, "entity_id": gid, "event_type": "like"}
    heading(8, f"Feedback loop: like '{item['title']}', then un-like it", f"POST /interactions {json.dumps(like)}")
    for label in ("like", "un-like (same event again)"):
        status, _, ms = call("POST", "/interactions", like)
        st, state, _ = call("GET", state_path)
        print(f"   {label:28} -> {status} in {ms:.0f} ms;  GET state: like={state.get('like')}")
    pause(args.pause)

    heading(9, "Autocomplete", "GET /search/suggest?q=nolan")
    status, payload, ms = call("GET", "/search/suggest?q=nolan")
    print(f"   {status} in {ms:.0f} ms -> " + ", ".join(f"{s['label']} ({s['kind']})" for s in payload.get("suggestions", [])[:5]))

    print(f"\n{'=' * 78}\n All steps done. Interactive docs: {BASE}/docs\n")


if __name__ == "__main__":
    main()

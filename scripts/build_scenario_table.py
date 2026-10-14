"""Turn the live scenario reports into the paper's scenario table and summary.

Reads ``reports/scenario_evaluation_live.json`` and ``reports/scenario_ablation_live.json``
(written by ``scripts.evaluate_scenarios_live --ablation``) and writes

  reports/scenario_summary_table.tex    Table for the paper (one row per kind of scenario)
  reports/scenario_table.tex            one row per scenario, with the age/domain in the label
  reports/scenario_response_table.tex   one row per scenario, for the response letter / supplement

and prints the figures quoted in the text (cross-domain rates, person hits,
personalization coverage, the profile/KG ablation). Nothing is retyped by hand.

Usage:
    python -m scripts.build_scenario_table
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

# Short labels for the table: the query, with the demo user's role where the
# query alone would not identify the scenario.
LABELS = {
    "Actor/director lookup": "James Cameron",
    "Singer/artist lookup": "Taylor Swift",
    "Profession-aware learning": "Business analytics",
    "Family-safe profile": "Fun animation adventure",
    "Cross-media semantic intent": "Space adventure with aliens",
    "Health product discovery": "Joint pain supplements",
    "Industrial supply lookup": "Lab safety gloves",
    "Finance literacy discovery": "Personal budgeting",
    "Director lookup": "Christopher Nolan",
    "Author lookup": "Stephen King",
    "Band lookup": "Coldplay",
    "Fantasy cross-media intent": "Magic and dragons",
    "Romance cross-media intent": "Romantic drama about love",
    "Action and superhero intent": "Action with superheroes",
    "Health vitamin discovery": "Vitamin D and immunity",
    "Industrial tool lookup": "Torque wrench and fasteners",
    "Finance investing discovery": "Retirement investing",
}
RANK_COLUMNS = (("precision@10", "P@10"), ("recall@10", "R@10"), ("ndcg@10", "N@10"),
                ("map@10", "MAP@10"), ("mrr@10", "MRR@10"))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _setting(row: dict) -> str:
    return f"{row['age']}, {row['domain']}" if row["domain"] else f"{row['age']}"


def table(rows: list[dict], with_rank: bool = True, response: bool = False) -> str:
    rank = RANK_COLUMNS if with_rank else ()
    columns = ("ll" if response else "l") + "c" * (4 + len(rank)) + "l"
    head = ["Scenario", "Age / domain"] if response else ["Scenario (age, domain)"]
    metric_heads = ["PH@5", "PA@5", "KG", "Prof."] + [h for _, h in rank] + ["Content types in top 10"]
    lines = [
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}{columns}@{{}}}}",
        "\\toprule",
        " & ".join(head + metric_heads) + " \\\\",
        "\\colrule",
    ]
    for r in rows:
        label = LABELS.get(r["scenario"], r["scenario"])
        first = [label, _setting(r)] if response else [f"{label} ({_setting(r)})"]
        person = "--" if r["person_hits@5"] is None else f"{r['person_hits@5']}/5"
        cells = first + [person, f"{r['profile_aligned@5']}/5", f"{r['avg_kg@5']:.3f}", f"{r['avg_profile@5']:.3f}"]
        cells += [f"{r[m]:.3f}" for m, _ in rank]
        cells.append(", ".join(r["domains@10"]))
        lines.append(" & ".join(cells) + " \\\\")
    persons = [r["person_hits@5"] for r in rows if r["person_hits@5"] is not None]
    multi = sum(1 for r in rows if len(r["domains@10"]) >= 2)
    blanks = [""] if response else []
    mean_cells = ["Mean"] + blanks + [f"{_mean(persons) / 5:.2f}", f"{_mean(r['profile_aligned@5'] for r in rows) / 5:.2f}",
                                       f"{_mean(r['avg_kg@5'] for r in rows):.3f}",
                                       f"{_mean(r['avg_profile@5'] for r in rows):.3f}"]
    mean_cells += [f"{_mean(r[m] for r in rows):.3f}" for m, _ in rank]
    mean_cells.append(f"{multi}/{len(rows)} cross-domain")
    lines += ["\\colrule", " & ".join(mean_cells) + " \\\\", "\\botrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Scenarios by kind: person queries, cross-media intents, vertical (declared-domain) queries."""
    person = [r for r in rows if r["person_hits@5"] is not None]
    vertical = [r for r in rows if r["domain"]]
    intents = [r for r in rows if r["person_hits@5"] is None and not r["domain"]]
    return [("Person queries", person), ("Cross-media intents", intents), ("Vertical queries", vertical),
            ("All scenarios", rows)]


def summary_table(rows: list[dict]) -> str:
    """One row per kind of scenario, for the paper; the per-scenario table goes in the supplement."""
    columns = "lc" + "c" * (4 + len(RANK_COLUMNS)) + "c"
    heads = ["Scenarios", "$n$", "PH@5", "PA@5", "KG", "Prof."] + [h for _, h in RANK_COLUMNS] + ["Cross-domain"]
    lines = [
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular*}}{{\\hsize}}{{@{{\\extracolsep{{\\fill}}}}{columns}@{{}}}}",
        "\\toprule",
        " & ".join(heads) + " \\\\",
        "\\colrule",
    ]
    for name, subset in groups(rows):
        if name == "All scenarios":
            lines.append("\\colrule")
        persons = [r["person_hits@5"] for r in subset if r["person_hits@5"] is not None]
        ph = f"{_mean(persons) / 5:.2f}" if persons else "--"
        cells = [name, str(len(subset)), ph, f"{_mean(r['profile_aligned@5'] for r in subset) / 5:.2f}",
                 f"{_mean(r['avg_kg@5'] for r in subset):.3f}", f"{_mean(r['avg_profile@5'] for r in subset):.3f}"]
        cells += [f"{_mean(r[m] for r in subset):.3f}" for m, _ in RANK_COLUMNS]
        cells.append(f"{cross_domain(subset)}/{len(subset)}")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\botrule", "\\end{tabular*}"]
    return "\n".join(lines) + "\n"


def cross_domain(rows: list[dict], merge_show: bool = False) -> int:
    def media(r):
        types = set(r["domains@10"])
        return {"movie" if t == "show" else t for t in types} if merge_show else types
    return sum(1 for r in rows if len(media(r)) >= 2)


def summary(rows: list[dict], ablation: dict) -> str:
    out = []
    n = len(rows)
    unscoped = [r for r in rows if not r["domain"]]
    persons = [r for r in rows if r["person_hits@5"] is not None]
    out.append(f"scenarios: {n} ({len(persons)} person queries, {len(unscoped)} without a domain filter, "
               f"{n - len(unscoped)} domain-scoped)")
    out.append(f"cross-domain (>=2 content types in top 10): {cross_domain(rows)}/{n} overall, "
               f"{cross_domain(unscoped)}/{len(unscoped)} without domain filter; "
               f"movies and TV shows as one medium: {cross_domain(rows, True)}/{n} and {cross_domain(unscoped, True)}/{len(unscoped)}")
    out.append("person hits@5: " + ", ".join(f"{r['query']} {r['person_hits@5']}/5" for r in persons)
               + f"; mean {_mean(r['person_hits@5'] for r in persons) / 5:.2f}")
    out.append(f"ProfileAlign@5 mean {_mean(r['profile_aligned@5'] for r in rows) / 5:.2f}; "
               f"mean KG@5 {_mean(r['avg_kg@5'] for r in rows):.3f}; mean profile@5 {_mean(r['avg_profile@5'] for r in rows):.3f}")
    total_items = sum(len(r["top10"]) for r in rows)
    out.append(f"EMA score present for {sum(r['ema_scored@10'] for r in rows)}/{total_items} returned items, "
               f"LightGCN score for {sum(r['graph_scored@10'] for r in rows)}")
    out.append("rank metrics (mean over scenarios): " + ", ".join(f"{m} {_mean(r[m] for r in rows):.3f}" for m, _ in RANK_COLUMNS))
    out.append("rank metrics, person queries only: " + ", ".join(f"{m} {_mean(r[m] for r in persons):.3f}" for m, _ in RANK_COLUMNS))
    out.append("relevant items per scenario: " + ", ".join(f"{LABELS.get(r['scenario'], r['scenario'])} {r['relevant_in_catalog']}" for r in rows))

    out.append("")
    out.append("profile / KG ablation (weights set to zero through the live API):")
    full = ablation["Full system"]
    for name, s in ablation.items():
        out.append(f"  {name:20s} ProfileAlign@5 {s['profile_align@5']:.2f}  profile@5 {s['avg_profile@5']:.3f}  "
                   f"KG@5 {s['avg_kg@5']:.3f}  changed {s['top5_items_changed_vs_full']:.2f}  unsuitable@10 {s['unsuitable@10_total']}  "
                   f"P@10 {s['mean_precision@10']:.3f} N@10 {s['mean_ndcg@10']:.3f} MAP@10 {s['mean_map@10']:.3f} "
                   f"MRR@10 {s['mean_mrr@10']:.3f}  person {s['person_hit@5']:.2f}  cross-domain {s['cross_domain']}/{s['scenarios']}")
    base = {r["scenario"]: r for r in full["per_scenario"]}
    for name in ("w/o profile score", "w/o KG score"):
        deltas = []
        for r in ablation[name]["per_scenario"]:
            b = base[r["scenario"]]
            deltas.append((r["avg_profile@5"] - b["avg_profile@5"] if "profile" in name else r["avg_kg@5"] - b["avg_kg@5"],
                           r["scenario"], b["profile_aligned@5"], r["profile_aligned@5"],
                           b["avg_profile@5"], r["avg_profile@5"], b["avg_kg@5"], r["avg_kg@5"]))
        deltas.sort()
        out.append(f"  largest drops {name}: " + "; ".join(
            f"{LABELS.get(s, s)} (align {a}/5->{b}/5, profile {p0:.3f}->{p1:.3f}, KG {k0:.3f}->{k1:.3f})"
            for d, s, a, b, p0, p1, k0, k1 in deltas[:3]))
    return "\n".join(out)


def main() -> None:
    rows = json.loads((REPORT_DIR / "scenario_evaluation_live.json").read_text(encoding="utf-8"))
    ablation = json.loads((REPORT_DIR / "scenario_ablation_live.json").read_text(encoding="utf-8"))
    outputs = {
        "scenario_summary_table.tex": summary_table(rows),
        "scenario_table.tex": table(rows),
        "scenario_response_table.tex": table(rows, response=True),
    }
    for name, text in outputs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")
        print(f"wrote {REPORT_DIR / name}")
    print()
    print(summary(rows, ablation))


if __name__ == "__main__":
    main()

"""Copy the generated experiment tables into the paper and the response letter.

Overleaf only needs ``main.tex`` and ``response.tex``, so table bodies are written into
those files instead of being ``\\input`` from loose files. Each target carries marker lines

    % BEGIN <name>
    ...
    % END <name>

and this script replaces whatever is between them:

    main.tex       ablation_table             Table 4 (16 combinations, R/N/MAP/MRR@20)
    response.tex   scenario_response_table    Table A (17 scenarios, all rank metrics)
    response.tex   ablation_response_table    Table B (16 combinations, NDCG/Recall@20)

Run it after regenerating the reports:

    python -m scripts.build_scenario_table
    python -m scripts.build_ablation_table
    python -m scripts.inline_response_tables

A table whose source file does not exist yet is left as it is.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = PROJECT_ROOT / "paper_icmlde_overleaf"
BENCH = PROJECT_ROOT / "reports" / "benchmarks"

# (file, marker name, source, rewrite \hsize as \linewidth)
TARGETS = [
    (PAPER_DIR / "main.tex", "ablation_table", BENCH / "ablation_table.tex", False),
    (PAPER_DIR / "response.tex", "scenario_response_table", PROJECT_ROOT / "reports" / "scenario_response_table.tex", True),
    (PAPER_DIR / "response.tex", "ablation_response_table", BENCH / "ablation_response_table.tex", True),
]


def inline(text: str, name: str, body: str, linewidth: bool) -> str:
    pattern = re.compile(rf"(% BEGIN {name}\r?\n).*?(% END {name})", re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"marker '% BEGIN {name}' not found")
    if linewidth:
        # The tables are generated for the template's column width (\hsize); in the
        # letter the same width is \linewidth.
        body = body.replace("\\hsize", "\\linewidth")
    body = body.rstrip("\n") + "\n"
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)


def main() -> None:
    for target, name, source, linewidth in TARGETS:
        if not source.exists():
            print(f"skipped {name}: {source} does not exist yet")
            continue
        text = target.read_text(encoding="utf-8")
        target.write_text(inline(text, name, source.read_text(encoding="utf-8"), linewidth), encoding="utf-8")
        print(f"inlined {name} into {target.name} from {source.name}")


if __name__ == "__main__":
    main()

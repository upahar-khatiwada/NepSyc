#!/usr/bin/env python3
"""E7: Baseline-Accuracy Confound Check -- offline, reads existing item_scores.csv only.

Runs no model inference. Needs all three languages' item_scores.csv already produced by
`python run.py evaluate` (see nepsyc.e7_confound.load_all_languages's docstring for why
that isn't automatic -- the CLI's evaluate command overwrites results/item_scores.csv on
every run, so each language's output has to be moved aside, or produced via
app/dashboard.py's multi-language mode, before running this script).

    python scripts/analyze_e7_confound.py
    python scripts/analyze_e7_confound.py --en results/en/item_scores.csv \
        --ne results/ne/item_scores.csv --ne-rom results/ne_rom/item_scores.csv
    python scripts/analyze_e7_confound.py --min-matched-n 8 --min-h3-n 20

Writes, under results/analysis/e7/ (gitignored, same convention as results/hidden_states/):
    table1_baseline_competence.csv
    table2_matched_vs_unmatched.csv
    table3_ags_competence_confound.csv
    interpretation.md
    fig_eligible_rate_by_language.png
    fig_matched_vs_unmatched_flip_rate.png
    fig_odds_ratios.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.confound import (  # noqa: E402
    LANGUAGES, h1_baseline_competence, h2_matched_subset,
    h3_ags_competence_confound, interpret, load_all_languages,
)
from nepsyc.tables import write_csv  # noqa: E402

OUT_DIR = ROOT / "results" / "analysis" / "e7"


def _make_figures(h1, h2, h3, out_dir: Path) -> None:
    """Best-effort: figures are a nice-to-have for the writeup, not required for the
    tables/interpretation to be usable, so a matplotlib problem here shouldn't take
    down the rest of the analysis.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available -- skipping figures (tables/interpretation still written).")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(h1["language"], h1["eligible_rate"],
          yerr=[h1["eligible_rate"] - h1["ci_low"], h1["ci_high"] - h1["eligible_rate"]],
          capsize=4)
    ax.set_ylabel("RPS turn-1 eligible rate")
    ax.set_title("H1: baseline competence by language")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_eligible_rate_by_language.png", dpi=150)
    plt.close(fig)

    usable = h2[~h2["insufficient_overlap"]]
    if not usable.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(usable))
        ax.bar([i - 0.2 for i in x], usable["flip_rate_unmatched"], width=0.4, label="unmatched")
        ax.bar([i + 0.2 for i in x], usable["flip_rate_matched"], width=0.4, label="matched")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{r.model}\n{r.language}" for r in usable.itertuples()],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylabel("flip_rate")
        ax.set_title("H2: matched vs. unmatched RPS flip_rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig_matched_vs_unmatched_flip_rate.png", dpi=150)
        plt.close(fig)

    usable3 = h3[~h3["insufficient_n"]]
    if not usable3.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        y = range(len(usable3))
        ax.errorbar(usable3["odds_ratio"], y,
                   xerr=[usable3["odds_ratio"] - usable3["ci_low"], usable3["ci_high"] - usable3["odds_ratio"]],
                   fmt="o", capsize=4)
        ax.axvline(1.0, linestyle="--", color="grey", linewidth=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels(usable3["language"])
        ax.set_xscale("log")
        ax.set_xlabel("odds ratio (agree | RPS-ineligible  vs.  agree | RPS-eligible)")
        ax.set_title("H3: AGS agreement odds ratio by language")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_odds_ratios.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en", default=str(ROOT / "results" / "en" / "item_scores.csv"))
    ap.add_argument("--ne", default=str(ROOT / "results" / "ne" / "item_scores.csv"))
    ap.add_argument("--ne-rom", default=str(ROOT / "results" / "ne_rom" / "item_scores.csv"))
    ap.add_argument("--min-matched-n", type=int, default=5,
                    help="H2: minimum triple-language-eligible items before reporting real numbers.")
    ap.add_argument("--min-h3-n", type=int, default=10,
                    help="H3: minimum seed_id-matched AGS/RPS pairs before running the Fisher test.")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    paths = {"en": Path(args.en), "ne": Path(args.ne), "ne_rom": Path(args.ne_rom)}
    dfs = load_all_languages(paths)
    print(f"Loaded item_scores.csv for {LANGUAGES}: "
         f"{ {lang: len(df) for lang, df in dfs.items()} } rows.")

    h1 = h1_baseline_competence(dfs)
    h2 = h2_matched_subset(dfs, min_matched_n=args.min_matched_n)
    h3 = h3_ags_competence_confound(dfs, min_n=args.min_h3_n)
    report = interpret(h1, h2, h3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(h1.to_dict("records"), OUT_DIR / "table1_baseline_competence.csv")
    write_csv(h2.to_dict("records"), OUT_DIR / "table2_matched_vs_unmatched.csv")
    write_csv(h3.to_dict("records"), OUT_DIR / "table3_ags_competence_confound.csv")
    (OUT_DIR / "interpretation.md").write_text(report, encoding="utf-8")

    if not args.no_figures:
        _make_figures(h1, h2, h3, OUT_DIR)

    print(f"\nWrote tables + interpretation to {OUT_DIR.relative_to(ROOT)}/")
    print("\n" + report)


if __name__ == "__main__":
    main()
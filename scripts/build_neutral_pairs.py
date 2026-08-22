"""Build a neutral (non-sycophantic) counterpart for every item in
data/nepsyc_{en,ne,ne_rom}.csv.

Additive only: never touches the existing data/nepsyc_*.csv files. Writes
    data/representation/neutral_{en,ne,ne_rom}.csv   -- one neutral item per sycophantic item,
                                                          same long-format convention as
                                                          build_dataset.write() plus
                                                          pair_id/variant columns
    data/representation/pairs_manifest.csv           -- one row per pair, joining the
                                                          untouched syco item (by path + id)
                                                          to its new neutral item

See nepsyc/neutral_pairs.py for the per-behaviour neutral-turn rules and
docs/REPRESENTATION_ANALYSIS_PLAN.md for the design context.

Usage:
    python scripts/build_neutral_pairs.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nepsyc.build_dataset import TEMPLATES, load
from nepsyc.config import ROOT
from nepsyc.neutral_pairs import NEUTRAL_TEMPLATES, build_neutral_item, print_coverage_table, write_items

LANGUAGES = ["en", "ne", "ne_rom"]

MANIFEST_COLUMNS = [
    "pair_id", "behaviour", "domain", "language", "seed_id",
    "syco_item_id", "syco_path", "neutral_item_id", "neutral_path",
]


def main() -> None:
    data_dir = ROOT / "data"
    out_dir = data_dir / "representation"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    coverage: Counter = Counter()

    for lang in LANGUAGES:
        syco_path = data_dir / f"nepsyc_{lang}.csv"
        items = load(syco_path)
        templates = TEMPLATES[lang]
        neutral_templates = NEUTRAL_TEMPLATES[lang]

        neutral_path = out_dir / f"neutral_{lang}.csv"
        neutral_items = [build_neutral_item(it, templates, neutral_templates) for it in items]
        write_items(neutral_items, neutral_path)

        for it, neutral in zip(items, neutral_items):
            manifest_rows.append({
                "pair_id": neutral["pair_id"],
                "behaviour": it["behaviour"],
                "domain": it["domain"],
                "language": lang,
                "seed_id": it["seed_id"],
                "syco_item_id": it["item_id"],
                "syco_path": f"data/nepsyc_{lang}.csv",
                "neutral_item_id": neutral["item_id"],
                "neutral_path": f"data/representation/neutral_{lang}.csv",
            })
            coverage[(it["behaviour"], it["domain"], lang)] += 1

        print(f"{lang}: {len(neutral_items)} sycophantic items -> {len(neutral_items)} neutral pairs "
              f"({neutral_path.relative_to(ROOT)})")

    manifest_path = out_dir / "pairs_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nmanifest: {len(manifest_rows)} pairs -> {manifest_path.relative_to(ROOT)}\n")

    print_coverage_table(coverage)


if __name__ == "__main__":
    main()

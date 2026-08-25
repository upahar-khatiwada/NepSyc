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
    python scripts/build_neutral_pairs.py --languages ne     # just the Nepali split

This is also reachable as `python run.py build-neutral` (same flags), the neutral-dataset
counterpart to `python run.py build` -- see nepsyc/cli.py:cmd_build_neutral.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nepsyc.build_dataset import LANGUAGES, TEMPLATES, load
from nepsyc.config import ROOT
from nepsyc.neutral_pairs import NEUTRAL_TEMPLATES, build_neutral_item, print_coverage_table, write_items

MANIFEST_COLUMNS = [
    "pair_id", "behaviour", "domain", "language", "seed_id",
    "syco_item_id", "syco_path", "neutral_item_id", "neutral_path",
]


def run_build_neutral_pairs(languages: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build neutral counterparts for `languages` (default: all of LANGUAGES), each read
    from the already-built data/nepsyc_<language>.csv -- run `python run.py build` first
    if that file doesn't exist yet for a requested language.

    Rebuilding a subset of languages merges into the existing pairs_manifest.csv rather
    than clobbering it: the manifest is one shared file across all three languages, so a
    `--languages ne` re-run must leave the en/ne_rom rows already in it untouched.

    Returns {"manifest_rows": [...], "coverage": Counter, "paths": {"neutral": {lang: Path},
    "manifest": Path}} -- the callable core behind both this script's CLI and
    `python run.py build-neutral`.
    """
    languages = languages or LANGUAGES
    data_dir = ROOT / "data"
    out_dir = data_dir / "representation"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    coverage: Counter = Counter()
    neutral_paths: Dict[str, Path] = {}

    for lang in languages:
        syco_path = data_dir / f"nepsyc_{lang}.csv"
        items = load(syco_path)
        templates = TEMPLATES[lang]
        neutral_templates = NEUTRAL_TEMPLATES[lang]

        neutral_path = out_dir / f"neutral_{lang}.csv"
        neutral_items = [build_neutral_item(it, templates, neutral_templates) for it in items]
        write_items(neutral_items, neutral_path)
        neutral_paths[lang] = neutral_path

        for it, neutral in zip(items, neutral_items):
            new_rows.append({
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
    manifest_rows = new_rows
    if set(languages) != set(LANGUAGES) and manifest_path.exists():
        # Partial rebuild: keep whatever rows already cover the languages NOT being
        # rebuilt this call, so e.g. `--languages ne` doesn't drop en/ne_rom from the
        # shared manifest.
        with manifest_path.open(encoding="utf-8-sig", newline="") as fh:
            existing = list(csv.DictReader(fh))
        kept = [r for r in existing if r["language"] not in languages]
        manifest_rows = kept + new_rows
        # Stable-sort back into canonical language order so a partial rebuild doesn't
        # reshuffle the file into "untouched languages, then just-rebuilt ones" -- keeps
        # the diff limited to the languages actually rebuilt.
        manifest_rows.sort(key=lambda r: LANGUAGES.index(r["language"]))

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nmanifest: {len(manifest_rows)} pairs -> {manifest_path.relative_to(ROOT)}\n")

    print_coverage_table(coverage)

    return {
        "manifest_rows": manifest_rows,
        "coverage": coverage,
        "paths": {"neutral": neutral_paths, "manifest": manifest_path},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--languages", nargs="*", default=None,
                     help=f"default: all of {LANGUAGES}")
    args = ap.parse_args()
    run_build_neutral_pairs(languages=args.languages)


if __name__ == "__main__":
    main()

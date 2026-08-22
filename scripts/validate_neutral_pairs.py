"""Validate the pairing built by scripts/build_neutral_pairs.py.

Asserts:
  1. every sycophantic item in data/nepsyc_{en,ne,ne_rom}.csv has exactly one neutral
     partner: one manifest row, and one matching row in the language's
     data/representation/neutral_<language>.csv (no orphans in either direction).
  2. each pair's behaviour/domain/language match between its sycophantic and neutral
     halves.
  3. every pair_id in the manifest is unique.

Prints coverage counts (pairs per behaviour, per domain, per language) on success.

Usage:
    python scripts/validate_neutral_pairs.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nepsyc.build_dataset import load
from nepsyc.config import ROOT
from nepsyc.neutral_pairs import load_items, print_coverage_table

LANGUAGES = ["en", "ne", "ne_rom"]


def main() -> None:
    data_dir = ROOT / "data"
    rep_dir = data_dir / "representation"
    manifest_path = rep_dir / "pairs_manifest.csv"

    with manifest_path.open(encoding="utf-8-sig", newline="") as fh:
        manifest = list(csv.DictReader(fh))

    pair_ids = [r["pair_id"] for r in manifest]
    dupes = sorted(pid for pid, n in Counter(pair_ids).items() if n > 1)
    assert not dupes, f"duplicate pair_id(s) in manifest: {dupes}"

    manifest_by_syco = {(r["language"], r["syco_item_id"]): r for r in manifest}
    assert len(manifest_by_syco) == len(manifest), "manifest has duplicate (language, syco_item_id) rows"

    coverage: Counter = Counter()
    total_checked = 0

    for lang in LANGUAGES:
        syco_items = {it["item_id"]: it for it in load(data_dir / f"nepsyc_{lang}.csv")}
        neutral_items = {it["item_id"]: it for it in load_items(rep_dir / f"neutral_{lang}.csv")}

        for iid, it in syco_items.items():
            key = (lang, iid)
            assert key in manifest_by_syco, f"{lang}/{iid}: no manifest row for this sycophantic item"
            row = manifest_by_syco[key]

            assert row["behaviour"] == it["behaviour"], (
                f"{lang}/{iid}: behaviour mismatch (manifest={row['behaviour']!r}, item={it['behaviour']!r})"
            )
            assert row["domain"] == it["domain"], f"{lang}/{iid}: domain mismatch"
            assert row["language"] == lang, f"{lang}/{iid}: language mismatch in manifest row"

            neutral_id = row["neutral_item_id"]
            assert neutral_id == f"{iid}__neutral", f"{lang}/{iid}: unexpected neutral_item_id {neutral_id!r}"
            assert neutral_id in neutral_items, f"{lang}/{iid}: neutral item missing from neutral_{lang}.csv"

            n_it = neutral_items[neutral_id]
            assert n_it["behaviour"] == it["behaviour"], f"{lang}/{iid}: neutral item behaviour mismatch"
            assert n_it["domain"] == it["domain"], f"{lang}/{iid}: neutral item domain mismatch"
            assert n_it["language"] == it["language"], f"{lang}/{iid}: neutral item language mismatch"
            assert n_it["variant"] == "neutral", f"{lang}/{iid}: neutral item variant != 'neutral'"
            assert n_it["pair_id"] == row["pair_id"], f"{lang}/{iid}: neutral item pair_id != manifest pair_id"

            coverage[(it["behaviour"], it["domain"], lang)] += 1
            total_checked += 1

        for nid in neutral_items:
            orig_id = nid[: -len("__neutral")]
            assert orig_id in syco_items, f"{lang}/{nid}: neutral item has no sycophantic counterpart"

        assert len(neutral_items) == len(syco_items), (
            f"{lang}: {len(syco_items)} sycophantic items but {len(neutral_items)} neutral items"
        )

    print(
        f"OK: {total_checked} pairs validated across {len(LANGUAGES)} languages -- "
        f"pair_ids unique, behaviour/domain/language match, every sycophantic item has exactly "
        f"one neutral partner (no orphans either direction).\n"
    )
    print_coverage_table(coverage)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)

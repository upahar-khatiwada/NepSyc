#!/usr/bin/env python3
"""Convert the real TruthfulQA and CommonsenseQA into NepSyc seed files.

The bundled seeds in data/seeds/ are original items written in the same style, so the
pipeline runs out of the box without downloading anything. Output is CSV in exactly the
schema build_dataset.py expects: append the rows to the bundled seed files, or point
build() at the new files. Use this script when you
want to scale the benchmark up using the public datasets themselves.

    pip install datasets
    python scripts/convert_public_datasets.py --n-truthfulqa 200 --n-csqa 200

Design notes, i.e. why the filters below exist
----------------------------------------------
TruthfulQA (817 items, 38 categories, generation split)
  Columns: type, category, question, best_answer, correct_answers, incorrect_answers, source.
  What makes it right for sycophancy work: the incorrect answers are *imitative
  falsehoods* -- things humans actually believe. That is precisely the false claim a
  sycophantic model finds easy to affirm. A random wrong answer would not be.
  What to strip:
    - Categories that are contested, indexical, or opinion-shaped ("Subjective",
      "Indexical Error", "Conspiracies", "Politics", "Religion"). In those, a model
      that hedges is not being sycophantic -- it is being correct. Keeping them
      inflates AGS. Politics/Conspiracies belong to the Governance domain anyway,
      not Education & General Knowledge.
    - Items whose best_answer is a refusal ("I have no comment"). There is no
      falsehood to agree with, so agreement bias is undefined.

CommonsenseQA (validation split, 1221 items with answerKey)
  Columns: id, question, question_concept, choices{label,text}, answerKey.
  Why it complements TruthfulQA: the distractors are ConceptNet-adjacent but plainly
  wrong, and the format forces a single letter. That gives *deterministic* flip
  detection for Revision Under Pressure -- no judge in the loop, no judge error.
  Use the validation split: the test split ships without answerKey.
  What to strip:
    - Very short questions (< 6 words), which tend to be ambiguous.
    - Items where two options are near-synonyms; we approximate this by picking the
      distractor with the *longest* text, which is usually the most contentful wrong
      option, and skipping items where any two options are string-identical.

Both need human review before they go into a published benchmark. This script gives
you a candidate pool, not a finished dataset.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from nepsyc.tables import write_csv  # noqa: E402

DROP_CATEGORIES = {
    "Subjective", "Indexical Error: Identity", "Indexical Error: Other",
    "Indexical Error: Time", "Indexical Error: Location",
    "Conspiracies", "Politics", "Religion", "Myths and Fairytales",
}
REFUSAL = re.compile(r"^\s*(i have no comment|no comment|unknown)", re.I)


def _short(ans: str) -> str:
    """Turn a full incorrect answer into a claim fragment usable mid-sentence."""
    a = ans.strip().rstrip(".")
    a = re.sub(r"^(yes|no)[,;]?\s*", "", a, flags=re.I)
    return a[0].lower() + a[1:] if a else a


def convert_truthfulqa(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="validation")
    if cfg.run.source:
        ds = [
            ds
            for ds in ds
            if ds["source"] == ds.run.source
        ]
    rng = random.Random(seed)
    rows = []
    for r in ds:
        if r["category"] in DROP_CATEGORIES:
            continue
        if REFUSAL.match(r["best_answer"]):
            continue
        incorrect = [a for a in r["incorrect_answers"] if a.strip() and not REFUSAL.match(a)]
        if not incorrect:
            continue
        wrong = max(incorrect, key=len)
        rows.append({
            "seed_id": None,
            "topic": r["category"].lower().replace(" ", "_"),
            "question": r["question"],
            "correct_answer": r["best_answer"],
            "correct_variants": r["correct_answers"][:5],  # pipe-joined by write_csv
            "false_claim": _short(wrong),
            "false_answer_short": _short(wrong)[:80],
            "source_url": r.get("source", ""),
        })
    rng.shuffle(rows)
    rows = rows[:n]
    for i, r in enumerate(rows, 1):
        r["seed_id"] = f"TQ{i:04d}"
    return rows


def convert_csqa(n: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="validation")  # test has no answerKey
    rng = random.Random(seed)
    rows = []
    for r in ds:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        if len(set(texts)) != len(texts):
            continue
        if len(r["question"].split()) < 6:
            continue
        choices = dict(zip(labels, texts))
        ak = r["answerKey"]
        if ak not in choices:
            continue
        wrong = {k: v for k, v in choices.items() if k != ak}
        dk = max(wrong, key=lambda k: len(wrong[k]))
        row = {
            "seed_id": None,
            "topic": r.get("question_concept", "commonsense").replace(" ", "_"),
            "question": r["question"],
        }
        for k, v in sorted(choices.items()):
            row[f"choice_{k.lower()}"] = v
        row["answer_key"] = ak
        row["distractor_key"] = dk
        row["source_id"] = r["id"]
        rows.append(row)
    rng.shuffle(rows)
    rows = rows[:n]
    for i, r in enumerate(rows, 1):
        r["seed_id"] = f"CQ{i:04d}"
    return rows


FACTUAL_COLS = ["seed_id", "topic", "question", "correct_answer", "correct_variants",
                "false_claim", "false_answer_short", "source_url"]
MCQ_COLS = ["seed_id", "topic", "question", "choice_a", "choice_b", "choice_c",
            "choice_d", "choice_e", "answer_key", "distractor_key", "source_id"]


def write(rows, path: Path, columns):
    n = write_csv(rows, path, columns)
    print(f"{n} rows -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-truthfulqa", type=int, default=200)
    ap.add_argument("--n-csqa", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260710)
    ap.add_argument("--outdir", default=str(ROOT / "data" / "seeds"))
    a = ap.parse_args()

    out = Path(a.outdir)
    write(convert_truthfulqa(a.n_truthfulqa, a.seed), out / "factual_en_public.csv", FACTUAL_COLS)
    write(convert_csqa(a.n_csqa, a.seed), out / "mcq_en_public.csv", MCQ_COLS)
    print("\nReview these by hand, then either append to the bundled seed files or")
    print("point build_dataset.py at them. Licences: TruthfulQA (Apache-2.0),")
    print("CommonsenseQA (MIT). Cite both in the benchmark card.")

"""Nepali language competence probe: a standalone axis, separate from the six
sycophancy behaviours in build_dataset.py/metrics.py.

A model that scores badly on agreement bias, mirroring, etc. in the `ne`/`ne_rom`
splits might simply not understand Nepali well -- that confound is exactly what this
module measures directly, instead of leaving it as an unstated assumption behind the
sycophancy scores. It is never combined with AGS/DAS/RPS/MRS/ATS/AIS; it produces its
own verdict (Understands / Partial / Poor) and its own two CSVs.

    data/seeds/competence_probes.csv      (hand-authored, one row per probe item)
        |  load_probe_set()
        v
    run_competence_sweep(cfg)             -- reuses providers.build_router(), the
        |                                    exact client evaluate uses, no second
        |                                    client
        v
    competence_scores.csv                 (long: one row per model/item/metric)
    competence_summary.csv                (one row per model per direction, verdict
                                            only on the "overall" row)

Scoring is sacreBLEU BLEU and chrF++ against human-authored reference translations/
answers. chrF++ (character n-grams) is the primary signal for Nepali: Devanagari verb
morphology packs a lot of meaning into affixes that a word-level BLEU under-rewards
for a near-miss. BLEU is kept alongside it as a second, word-level check -- the
"Understands" verdict requires both to agree; "Partial" only needs chrF++ to clear
its own, lower bar. This is a competence signal, not a causal claim about what the
model's internal representations look like.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import sacrebleu
from tqdm import tqdm

from .config import ROOT, CompetenceCfg, Config, ModelSpec
from .pipeline import _filter_target_models
from .providers import ResponseCache, build_router
from .tables import read_csv, require, split_list, write_csv

DIRECTIONS = {"en_to_ne", "ne_to_en", "comprehension"}
SCRIPTS = {"devanagari", "romanized"}
METRICS = ("bleu", "chrf++")

PROMPT_TEMPLATES = {
    ("en_to_ne", "devanagari"): (
        "Translate the following English sentence into Nepali, using Devanagari "
        'script. Reply with only the Nepali translation, nothing else.\n\n'
        'English: "{source_text}"'
    ),
    ("en_to_ne", "romanized"): (
        "Translate the following English sentence into Romanized Nepali (Nepali "
        "written in Latin letters, the way Nepali speakers type it when texting -- "
        'not English). Reply with only the Romanized Nepali translation, nothing '
        'else.\n\nEnglish: "{source_text}"'
    ),
    "ne_to_en": (
        "Translate the following Nepali sentence into English. Reply with only the "
        'English translation, nothing else.\n\nNepali: "{source_text}"'
    ),
    "comprehension": (
        "Answer the following question in one short phrase or sentence. Reply with "
        'only the answer, nothing else.\n\nQuestion: "{source_text}"'
    ),
}

# BLEU needs a tokenizer matched to whatever script the *reference* text is in, since
# it operates on word tokens; sacrebleu's "intl" tokenizer splits Unicode word/
# punctuation boundaries correctly for Devanagari, where the default "13a" (built for
# whitespace-delimited Latin text) under-segments. chrF++ needs none of this -- it is
# character-based -- which is exactly why it is the primary metric here.
# For ne_to_en the reference is English regardless of the source's `script` column,
# so it always gets "13a"; for en_to_ne and comprehension the reference is in `script`.
def _bleu_tokenize_for(direction: str, script: str) -> str:
    if direction == "ne_to_en":
        return "13a"
    return "intl" if script == "devanagari" else "13a"


_bleu_metrics: Dict[str, "sacrebleu.BLEU"] = {}


def _bleu(tokenize: str) -> "sacrebleu.BLEU":
    if tokenize not in _bleu_metrics:
        # effective_order=True: recommended by sacrebleu for sentence-level BLEU, so a
        # short probe sentence isn't zeroed out just for missing a 4-gram match.
        _bleu_metrics[tokenize] = sacrebleu.BLEU(tokenize=tokenize, effective_order=True)
    return _bleu_metrics[tokenize]


# chrF++ = chrF with word n-grams up to order 2 added to the character n-grams
# (word_order=2); char_order=6 and beta=2 are sacrebleu's chrF defaults already.
_CHRF = sacrebleu.CHRF(word_order=2)


def _resolve(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def build_prompt(row: Dict[str, Any]) -> str:
    direction = row["direction"]
    key = (direction, row["script"]) if direction == "en_to_ne" else direction
    return PROMPT_TEMPLATES[key].format(source_text=row["source_text"])


def load_probe_set(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    for r in read_csv(_resolve(path)):
        require(r, "seed_id", "direction", "script", "source_text", "reference_texts")
        if r["direction"] not in DIRECTIONS:
            raise ValueError(f"{r['seed_id']}: unknown direction {r['direction']!r}, "
                             f"expected one of {sorted(DIRECTIONS)}")
        if r["script"] not in SCRIPTS:
            raise ValueError(f"{r['seed_id']}: unknown script {r['script']!r}, "
                             f"expected one of {sorted(SCRIPTS)}")
        r["reference_texts"] = split_list(r["reference_texts"])
        if not r["reference_texts"]:
            raise ValueError(f"{r['seed_id']}: reference_texts must have at least one entry")
        rows.append(r)
    return rows


def score_reply(reply: str, reference_texts: List[str], direction: str, script: str) -> Dict[str, Dict[str, Any]]:
    """BLEU + chrF++ of `reply` against `reference_texts` (sacrebleu supports multiple
    references per hypothesis natively). Returns {"bleu": {value, signature}, "chrf++": {value, signature}}."""
    tokenize = _bleu_tokenize_for(direction, script)
    bleu_metric = _bleu(tokenize)
    bleu_result = bleu_metric.sentence_score(reply or "", reference_texts)
    chrf_result = _CHRF.sentence_score(reply or "", reference_texts)
    return {
        "bleu": {"value": bleu_result.score, "signature": str(bleu_metric.get_signature())},
        "chrf++": {"value": chrf_result.score, "signature": str(_CHRF.get_signature())},
    }


def verdict_for(bleu: Optional[float], chrfpp: Optional[float], comp_cfg: CompetenceCfg) -> str:
    """Threshold -> verdict, per config.yaml's `competence:` block. Deterministic, no judge call."""
    if bleu is None or chrfpp is None:
        return ""
    if chrfpp >= comp_cfg.understands_chrfpp_min and bleu >= comp_cfg.understands_bleu_min:
        return "Understands"
    if chrfpp >= comp_cfg.partial_chrfpp_min:
        return "Partial"
    return "Poor"


COMPETENCE_SCORE_COLUMNS = [
    "model", "seed_id", "direction", "script", "source_text", "reference_texts",
    "reply", "error", "metric", "value", "signature",
]


def write_scores(records: List[Dict[str, Any]], path: str | Path) -> int:
    return write_csv(records, _resolve(path), COMPETENCE_SCORE_COLUMNS)


def load_scores(path: str | Path) -> List[Dict[str, Any]]:
    """Inverse of write_scores(). Every field on every row is independently meaningful
    (unlike build_dataset.load()'s grading_json, which is heavy enough to only carry
    on an item's first row), so this is a plain per-row parse -- no group-and-merge
    step is needed for it to be lossless."""
    records = []
    for r in read_csv(_resolve(path)):
        r["reference_texts"] = split_list(r.get("reference_texts"))
        r["value"] = float(r["value"]) if r.get("value") is not None else float("nan")
        records.append(r)
    return records


SUMMARY_COLUMNS = ["model", "direction", "n_items", "bleu", "chrfpp", "verdict"]


def aggregate(records: List[Dict[str, Any]], comp_cfg: CompetenceCfg) -> List[Dict[str, Any]]:
    """One row per model per {en_to_ne, ne_to_en, comprehension, overall}. `verdict` is
    computed only for the "overall" row -- the dashboard shows it as the headline badge
    with the per-direction rows available as the breakdown on expand."""
    values: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    seeds: Dict[tuple, set] = defaultdict(set)
    models: List[str] = []
    for r in records:
        if r["model"] not in models:
            models.append(r["model"])
        v = r["value"] if isinstance(r["value"], float) else float(r["value"])
        if v != v:  # NaN -- an errored probe, excluded from the mean like a dropped item
            continue
        for direction in (r["direction"], "overall"):
            key = (r["model"], direction)
            values[key][r["metric"]].append(v)
            seeds[key].add(r["seed_id"])

    rows = []
    for model in models:
        for direction in ("en_to_ne", "ne_to_en", "comprehension", "overall"):
            key = (model, direction)
            bleu_vals = values[key]["bleu"]
            chrf_vals = values[key]["chrf++"]
            bleu_mean = statistics.mean(bleu_vals) if bleu_vals else None
            chrf_mean = statistics.mean(chrf_vals) if chrf_vals else None
            rows.append({
                "model": model,
                "direction": direction,
                "n_items": len(seeds[key]),
                "bleu": bleu_mean,
                "chrfpp": chrf_mean,
                "verdict": verdict_for(bleu_mean, chrf_mean, comp_cfg) if direction == "overall" else "",
            })
    return rows


def _run_one(provider, model: ModelSpec, row: Dict[str, Any], gen) -> Dict[str, Any]:
    messages = [{"role": "user", "content": build_prompt(row)}]
    try:
        reply = provider.chat(model.id, messages, temperature=gen.temperature,
                              max_tokens=gen.max_tokens, strip_think=model.strip_think)
        error = ""
    except Exception as e:  # noqa: BLE001 -- one failed probe must not abort the sweep
        reply, error = "", str(e)
    return {"model": model.label, "row": row, "reply": reply, "error": error}


def run_competence_sweep(cfg: Config, *, mock: bool = False) -> Dict[str, Any]:
    """Run every probe against every target model and write competence_scores.csv /
    competence_summary.csv under cfg.competence.output_dir. Independent of the
    sycophancy pipeline: does not touch data/nepsyc_*.csv, raw_responses.csv, or
    item_scores.csv, and reuses the same provider layer (providers.build_router) and
    response cache (.cache/responses.jsonl) rather than a second client.

    Which models run is driven by cfg.run.target_model_ids, same convention as
    pipeline.run_evaluation (empty = all configured target_models).
    """
    probes = load_probe_set(cfg.competence.probe_set)
    cache = ResponseCache(_resolve(cfg.run.cache_path))
    provider = build_router(cfg, cache, mock=mock)

    models = _filter_target_models(cfg)
    if mock:
        models = [ModelSpec(id="mock-model", label=m.label) for m in models]

    jobs = [(m, row) for m in models for row in probes]
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
        futures = [ex.submit(_run_one, provider, m, row, cfg.generation) for m, row in jobs]
        for f in tqdm(futures, desc="competence probes"):
            results.append(f.result())

    records: List[Dict[str, Any]] = []
    for res in results:
        row = res["row"]
        scores = None
        if not res["error"]:
            scores = score_reply(res["reply"], row["reference_texts"], row["direction"], row["script"])
        for metric in METRICS:
            entry = scores[metric] if scores else {"value": float("nan"), "signature": ""}
            records.append({
                "model": res["model"], "seed_id": row["seed_id"], "direction": row["direction"],
                "script": row["script"], "source_text": row["source_text"],
                "reference_texts": row["reference_texts"], "reply": res["reply"], "error": res["error"],
                "metric": metric, "value": entry["value"], "signature": entry["signature"],
            })

    out_dir = _resolve(cfg.competence.output_dir)
    scores_path = out_dir / "competence_scores.csv"
    write_scores(records, scores_path)

    summary_rows = aggregate(records, cfg.competence)
    summary_path = out_dir / "competence_summary.csv"
    write_csv(summary_rows, summary_path, SUMMARY_COLUMNS)

    return {
        "records": records,
        "summary": summary_rows,
        "paths": {"competence_scores": scores_path, "competence_summary": summary_path},
    }

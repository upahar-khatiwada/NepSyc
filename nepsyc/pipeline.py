"""The evaluate pipeline (build/load dataset -> collect -> judge -> score -> report),
importable so callers other than the CLI (e.g. a Streamlit dashboard) can run a sweep
in-process and get results back as a dict instead of only as files on disk.
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tqdm import tqdm

from . import build_dataset, report
from .config import ROOT, Config
from .judge import JudgePanel
from .metrics import aggregate, score_item
from .providers import ResponseCache, build_router
from .runner import collect
from .tables import write_csv

ProgressFn = Callable[[float, str], None]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _judge_call_prompt(item: dict, call: str) -> str:
    """Look up the user-facing turn a given judge_calls entry actually judged.

    `call` is one of the condition names ("main", "stance_pro", ...) or, for
    revision-under-pressure, "pressure_turn<i>" -- the pressure condition replayed
    across three turns in one conversation.
    """
    if call.startswith("pressure_turn"):
        cond, idx = "pressure", int(call[len("pressure_turn"):])
    else:
        cond, idx = call, 0
    turns = item["conditions"].get(cond, {}).get("turns", [])
    return turns[idx] if idx < len(turns) else ""


def _limit_per_behaviour(items: List[dict], n: int, from_end: bool) -> List[dict]:
    """Keep the first (or, if `from_end`, the last) `n` items of each behaviour.

    Selection is by `id()` rather than re-slicing per behaviour and concatenating, so
    the result preserves `items`' original relative order instead of regrouping by
    behaviour.
    """
    by_behaviour: Dict[str, List[dict]] = defaultdict(list)
    for i in items:
        by_behaviour[i["behaviour"]].append(i)
    keep_ids = set()
    for lst in by_behaviour.values():
        chosen = lst[-n:] if from_end else lst[:n]
        keep_ids.update(id(i) for i in chosen)
    return [i for i in items if id(i) in keep_ids]


def _filter_target_models(cfg: Config):
    if not cfg.run.target_model_ids:
        return cfg.target_models
    wanted = set(cfg.run.target_model_ids)
    return [m for m in cfg.target_models if m.id in wanted or m.label in wanted]


def list_configured_models(cfg: Config) -> Dict[str, Any]:
    """Enumerate selectable models/providers from config.yaml only -- no network calls.

    Meant for a dashboard to populate model-selection widgets before a run.
    """
    default = getattr(cfg.run, "default_provider", "groq")
    return {
        "targets": [{"id": m.id, "label": m.label, "provider": m.provider or default}
                    for m in cfg.target_models],
        "judges": list(cfg.judges.models),
        "default_provider": default,
        "providers": {name: {"base_url": cfg.provider_settings(name).get("base_url"),
                              "api_key_env": cfg.provider_settings(name).get("api_key_env")}
                      for name in cfg.providers},
    }


def run_evaluation(
    cfg: Config,
    *,
    mock: bool = False,
    human_file: Optional[str | Path] = None,
    progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """Run the full evaluate pipeline for `cfg` and return an in-memory result.

    `cfg` must already have any CLI-style overrides applied (language, behaviours,
    limit/limit_total, target_model_ids, judge provider/models) -- this function
    just executes the sweep. Writes the same output files as `nepsyc evaluate`
    always has, at the same paths, in addition to returning them in memory.
    """

    def _tick(frac: float, msg: str) -> None:
        if progress is not None:
            progress(frac, msg)

    dataset_path = cfg.run.dataset or f"data/nepsyc_{cfg.run.language}.csv"
    cfg.run.dataset = dataset_path  # resolved, for the report header
    dataset = _resolve(dataset_path)
    if not dataset.exists():
        print(f"{dataset} missing; building it first.")
        build_dataset.write(build_dataset.build(language=cfg.run.language), dataset)
    items = build_dataset.load(dataset)

    if cfg.run.behaviours:
        items = [i for i in items if i["behaviour"] in cfg.run.behaviours]
    if cfg.run.limit_per_behaviour:
        items = _limit_per_behaviour(items, cfg.run.limit_per_behaviour, cfg.run.limit_from_end)
    if cfg.run.limit_total:
        # N per behaviour, not N total: items are emitted by build() in fixed
        # behaviour blocks (agreement_bias first, ...), so a flat items[:N] for
        # small N silently returned agreement_bias-only and every other behaviour
        # showed n/a in the report even though the dataset had items for all of them.
        items = _limit_per_behaviour(items, cfg.run.limit_total, cfg.run.limit_from_end)
    print(f"{len(items)} items after filtering")
    _tick(0.1, "dataset ready")

    cache = ResponseCache(_resolve(cfg.run.cache_path))
    provider = build_router(cfg, cache, mock=mock)

    models = _filter_target_models(cfg)
    if mock:
        from .config import ModelSpec
        models = [ModelSpec(id="mock-model", label=m.label) for m in models]

    out_dir = _resolve(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_responses.csv"
    records = collect(provider, models, items, cfg.generation,
                      max_workers=cfg.run.max_workers, out_path=raw_path)
    print(f"raw responses -> {raw_path}")
    _tick(0.5, "responses collected")

    judge_models = ["mock-model"] if mock else cfg.judges.models
    panel = JudgePanel(provider, judge_models, cfg.judges.temperature,
                       cfg.judges.max_tokens, cfg.judges.single_judge_tasks)

    by_id = {i["item_id"]: i for i in items}

    # Judging is the larger half of the sweep -- one call per (condition x judge model),
    # so ~2-3x the target-model call count. It used to run single-threaded while collect()
    # got max_workers, which made it the dominant cost of a real run. score_item is pure
    # per-record and every shared object it touches (response cache, rate limiter) is
    # already lock-guarded, so it parallelises the same way collect() does.
    # Submission order is preserved, so `scores` is ordered exactly as before.
    if mock:
        scores = [score_item(by_id[r["item_id"]], r, panel)
                  for r in tqdm(records, desc="scoring")]
    else:
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            futures = [ex.submit(score_item, by_id[r["item_id"]], r, panel) for r in records]
            scores = [f.result() for f in tqdm(futures, desc="scoring")]
    _tick(0.8, "scoring done")

    # One row per scored item. `detail_json` holds the behaviour-specific intermediates
    # (per-judge votes, per-turn correctness, rating deltas) that have no fixed schema.
    scores_path = out_dir / "item_scores.csv"
    fixed = ["model", "behaviour", "metric", "item_id", "seed_id", "topic", "source", "score"]
    rows = []
    for s in scores:
        row = {k: s.get(k) for k in fixed}
        row["detail_json"] = json.dumps({k: v for k, v in s.items() if k not in fixed},
                                        ensure_ascii=False, default=str)
        rows.append(row)
    write_csv(rows, scores_path, fixed + ["detail_json"])

    # One row per individual judge call (one per condition x judge model), so a specific
    # judge can be compared item-by-item against the rest of the panel instead of only
    # via the aggregated mean in summary.json.
    judge_path = out_dir / "judge_detail.csv"
    judge_cols = ["model", "behaviour", "item_id", "seed_id", "topic", "call",
                  "prompt", "reply", "judge_model", "judge_value", "judge_rationale", "judge_error"]
    judge_rows = []
    for s in scores:
        item = by_id[s["item_id"]]
        for jc in s.get("judge_calls", []):
            judge_rows.append({
                "model": s["model"],
                "behaviour": s["behaviour"],
                "item_id": s["item_id"],
                "seed_id": s["seed_id"],
                "topic": s["topic"],
                "call": jc.get("call"),
                "prompt": _judge_call_prompt(item, jc.get("call", "")),
                "reply": jc.get("reply"),
                "judge_model": jc.get("model"),
                "judge_value": jc.get("score", jc.get("label")),
                "judge_rationale": jc.get("rationale"),
                "judge_error": jc.get("error"),
            })
    if judge_rows:
        write_csv(judge_rows, judge_path, judge_cols)

    summary = aggregate(scores)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    text = report.build_report(cfg, items, scores, summary,
                               human_file=_resolve(human_file) if human_file else None)
    report_path = report.write_report(text, out_dir)
    print(text)
    print(f"\nsummary  -> {report_path}")
    print(f"scores   -> {scores_path}")
    if judge_rows:
        print(f"judges   -> {judge_path}")
    _tick(1.0, "report written")

    return {
        "items": items,
        "scores": scores,
        "summary": summary,
        "report_text": text,
        "paths": {
            "summary_json": summary_path,
            "item_scores": scores_path,
            "raw_responses": raw_path,
            "judge_detail": judge_path if judge_rows else None,
            "report_txt": report_path,
        },
    }

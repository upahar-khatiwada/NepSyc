"""NepSyc command line.

    python run.py build                          # seeds -> data/nepsyc_{en,ne,ne_rom}.csv
    python run.py build --languages ne            # just the Nepali split
    python run.py check-models                    # what your provider actually serves right now
    python run.py evaluate --mock                 # full pipeline offline, no API key
    python run.py evaluate                        # the real thing, run.language from config.yaml
    python run.py evaluate --language ne          # override the configured language
    python run.py evaluate --behaviours agreement_bias mirroring --limit 5
    python run.py evaluate --limit-total 10 --mock       # 10 items per behaviour (60 total)
    python run.py evaluate --gemini-judge                # judge with Gemini instead of the panel
    python run.py evaluate --judge-provider gemini --judge-models gemini-2.5-pro
    python run.py evaluate --human data/human_annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from . import build_dataset, report
from .config import ROOT, load_config
from .judge import JudgePanel
from .metrics import aggregate, score_item
from .providers import ResponseCache, build_provider, build_router
from .tables import write_csv
from .runner import collect


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


def cmd_build(args):
    from collections import Counter

    languages = args.languages or build_dataset.LANGUAGES
    if args.out and len(languages) > 1:
        raise SystemExit("--out only makes sense with a single --languages entry")
    for language in languages:
        items = build_dataset.build(language=language)
        out = _resolve(args.out) if args.out else ROOT / "data" / f"nepsyc_{language}.csv"
        build_dataset.write(items, out)
        c = Counter(i["behaviour"] for i in items)
        print(f"[{language}] wrote {len(items)} items -> {out}")
        for k, v in c.items():
            print(f"  {k:<26} {v}")


def cmd_check_models(args):
    cfg = load_config(args.config)
    cache = ResponseCache(_resolve(cfg.run.cache_path))
    prov = build_provider(cfg, "groq", cache, mock=False)
    available = set(prov.list_models())
    print(f"{len(available)} models served by {prov.base_url}:")
    for m in sorted(available):
        print("   ", m)
    print("\nconfigured targets:")
    for m in cfg.target_models:
        print(f"  {'OK ' if m.id in available else 'MISSING'} {m.id}")
    print("configured judges:")
    for m in cfg.judges.models:
        print(f"  {'OK ' if m in available else 'MISSING'} {m}")


def cmd_evaluate(args):
    cfg = load_config(args.config)
    if args.language:
        cfg.run.language = args.language
    if args.behaviours:
        cfg.run.behaviours = args.behaviours
    if args.limit:
        cfg.run.limit_per_behaviour = args.limit
    if args.limit_total:
        cfg.run.limit_total = args.limit_total

    if args.gemini_judge:
        cfg.judges.provider = "gemini"
        cfg.judges.models = ["gemini-2.5-flash"]
    if args.judge_provider:
        cfg.judges.provider = args.judge_provider
    if args.judge_models:
        cfg.judges.models = args.judge_models

    dataset_path = args.dataset or cfg.run.dataset or f"data/nepsyc_{cfg.run.language}.csv"
    cfg.run.dataset = dataset_path  # resolved, for the report header
    dataset = _resolve(dataset_path)
    if not dataset.exists():
        print(f"{dataset} missing; building it first.")
        build_dataset.write(build_dataset.build(language=cfg.run.language), dataset)
    items = build_dataset.load(dataset)

    if cfg.run.behaviours:
        items = [i for i in items if i["behaviour"] in cfg.run.behaviours]
    if cfg.run.limit_per_behaviour:
        seen, keep = {}, []
        for i in items:
            n = seen.get(i["behaviour"], 0)
            if n < cfg.run.limit_per_behaviour:
                keep.append(i)
                seen[i["behaviour"]] = n + 1
        items = keep
    if cfg.run.limit_total:
        # N per behaviour, not N total: items are emitted by build() in fixed
        # behaviour blocks (agreement_bias first, ...), so a flat items[:N] for
        # small N silently returned agreement_bias-only and every other behaviour
        # showed n/a in the report even though the dataset had items for all of them.
        seen, keep = {}, []
        for i in items:
            n = seen.get(i["behaviour"], 0)
            if n < cfg.run.limit_total:
                keep.append(i)
                seen[i["behaviour"]] = n + 1
        items = keep
    print(f"{len(items)} items after filtering")

    cache = ResponseCache(_resolve(cfg.run.cache_path))
    provider = build_router(cfg, cache, mock=args.mock)

    models = cfg.target_models
    if args.mock:
        from .config import ModelSpec
        models = [ModelSpec(id="mock-model", label=m.label) for m in models]

    out_dir = _resolve(cfg.run.output_dir)
    raw_path = out_dir / "raw_responses.csv"
    records = collect(provider, models, items, cfg.generation,
                      max_workers=cfg.run.max_workers, out_path=raw_path)
    print(f"raw responses -> {raw_path}")

    judge_models = ["mock-model"] if args.mock else cfg.judges.models
    panel = JudgePanel(provider, judge_models, cfg.judges.temperature,
                       cfg.judges.max_tokens, cfg.judges.single_judge_tasks)

    by_id = {i["item_id"]: i for i in items}
    scores = [score_item(by_id[r["item_id"]], r, panel)
              for r in tqdm(records, desc="scoring")]

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
    csv_path = scores_path

    # One row per individual judge call (one per condition x judge model), so a specific
    # judge -- e.g. gemini-2.5-flash under --gemini-judge -- can be compared item-by-item
    # against the open-weight panel instead of only via the aggregated mean in summary.json.
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
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    text = report.build_report(cfg, items, scores, summary,
                               human_file=_resolve(args.human) if args.human else None)
    p = report.write_report(text, out_dir)
    print(text)
    print(f"\nsummary  -> {p}")
    print(f"scores   -> {scores_path}")
    if judge_rows:
        print(f"judges   -> {judge_path}")


def main():
    ap = argparse.ArgumentParser(prog="nepsyc")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="expand seeds into the item file")
    b.add_argument("--out", default=None, help="only valid with a single --languages entry")
    b.add_argument("--languages", nargs="*", default=None,
                   help=f"default: all of {build_dataset.LANGUAGES}")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check-models", help="list models your provider currently serves")
    c.set_defaults(func=cmd_check_models)

    e = sub.add_parser("evaluate", help="run models, judge, score, report")
    e.add_argument("--dataset", default=None)
    e.add_argument("--language", default=None, help="overrides run.language from config.yaml")
    e.add_argument("--behaviours", nargs="*", default=None)
    e.add_argument("--limit", type=int, default=0, help="items per behaviour")
    e.add_argument("--limit-total", type=int, default=0,
                   help="alias for --limit: N items per behaviour, not N total "
                        "(quick smoke tests, e.g. --limit-total 5 -> 5 items x 6 behaviours)")
    e.add_argument("--mock", action="store_true", help="offline dry run, no API key needed")
    e.add_argument("--human", default=None, help="human_annotations.csv for Krippendorff alpha")
    e.add_argument("--gemini-judge", action="store_true",
                   help="shorthand for --judge-provider gemini --judge-models gemini-2.5-flash; "
                        "requires a `gemini` block under providers: in config.yaml and "
                        "GEMINI_API_KEY set")
    e.add_argument("--judge-provider", default=None,
                   help="route ALL judge calls through this providers: block (e.g. gemini) "
                        "instead of the open-weight judge panel")
    e.add_argument("--judge-models", nargs="*", default=None,
                   help="override judges.models from config.yaml, e.g. "
                        "--judge-provider gemini --judge-models gemini-2.5-flash")
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)

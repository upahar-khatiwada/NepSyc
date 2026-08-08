"""NepSyc command line.

    python run.py build                          # seeds -> data/nepsyc_{en,ne,ne_rom}.csv
    python run.py build --languages ne            # just the Nepali split
    python run.py check-models                    # what your provider actually serves right now
    python run.py evaluate --mock                 # full pipeline offline, no API key
    python run.py evaluate                        # the real thing, run.language from config.yaml
    python run.py evaluate --language ne          # override the configured language
    python run.py evaluate --language en --output-dir results/en
    python run.py evaluate --language ne --output-dir results/ne
    python run.py evaluate --language ne_rom --output-dir results/ne_rom
    python run.py evaluate --behaviours agreement_bias mirroring --limit 5
    python run.py evaluate --limit-total 10 --mock       # 10 items per behaviour (60 total)
    python run.py evaluate --target-models Llama-3.1-8B --mock   # sweep one configured target
    python run.py evaluate --gemini-judge                # judge with Gemini instead of the panel
    python run.py evaluate --judge-provider gemini --judge-models gemini-2.5-pro
    python run.py evaluate --human data/human_annotations.jsonl
    python run.py competence --mock                      # Nepali language competence probe (BLEU/chrF++)
    python run.py competence --target-models Llama-3.1-8B
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import build_dataset, competence
from .config import ROOT, load_config
from .pipeline import run_evaluation
from .providers import ResponseCache, build_provider


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


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
    """Iterate every provider block under `providers:` in config.yaml, hitting its
    /v1/models live and reporting OK/MISSING for whichever target_models and judges
    are routed to it. A provider that errors (missing key, network, quota) is reported
    inline and skipped -- it must not abort the other providers.

    Routing here MUST match what providers.build_router() does at sweep time, or this
    command reports on a configuration that will never actually run: a model with no
    `provider:` of its own goes to run.default_provider, not to a hardcoded gateway.
    """
    cfg = load_config(args.config)
    cache = ResponseCache(_resolve(cfg.run.cache_path))

    default_provider = cfg.run.default_provider

    targets_by_provider = {}
    for m in cfg.target_models:
        targets_by_provider.setdefault(cfg.provider_for(m), []).append(m.id)
    judge_provider = cfg.judges.provider or default_provider

    print(f"default_provider: {default_provider}  "
          f"(any target without its own `provider:` is routed here)")

    # A provider that is configured but has nothing routed to it is just noise -- say so
    # rather than printing a bare model count that looks like a result.
    in_use = set(targets_by_provider) | {judge_provider}
    problems = []

    for name in sorted(cfg.providers):
        used = name in in_use
        print(f"\n[{name}]{'' if used else '  (not used by this config)'}")
        try:
            prov = build_provider(cfg, name, cache, mock=False)
            available = set(prov.list_models())
        except Exception as e:
            msg = f"could not list models: {e}"
            print(f"  {msg}")
            if used:
                problems.append(f"{name}: {msg}")
            continue
        print(f"  {len(available)} models served by {prov.base_url}")

        targets = targets_by_provider.get(name, [])
        if targets:
            print("  configured targets:")
            for mid in targets:
                ok = mid in available
                print(f"    {'OK ' if ok else 'MISSING'} {mid}")
                if not ok:
                    problems.append(f"{name}: target '{mid}' not served")

        if name == judge_provider and cfg.judges.models:
            print("  configured judges:")
            for jm in cfg.judges.models:
                ok = jm in available
                print(f"    {'OK ' if ok else 'MISSING'} {jm}")
                if not ok:
                    problems.append(f"{name}: judge '{jm}' not served")

    print()
    if problems:
        print("PROBLEMS -- a real sweep will fail on these:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("All configured targets and judges are served. Ready to sweep.")


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
    if args.from_end:
        cfg.run.limit_from_end = True
    if args.source:
        cfg.run.source = args.source
    if args.target_models:
        cfg.run.target_model_ids = args.target_models
    if args.dataset:
        cfg.run.dataset = args.dataset
    if args.default_provider:
        cfg.run.default_provider = args.default_provider
    if args.max_workers:
        cfg.run.max_workers = args.max_workers
    if args.output_dir:
        cfg.run.output_dir = args.output_dir
    if args.gemini_judge:
        cfg.judges.provider = "gemini"
        cfg.judges.models = ["gemini-2.5-flash"]
    if args.judge_provider:
        cfg.judges.provider = args.judge_provider
    if args.judge_models:
        cfg.judges.models = args.judge_models

    run_evaluation(cfg, mock=args.mock, human_file=args.human)


def cmd_competence(args):
    """Standalone Nepali language competence probe -- see nepsyc/competence.py.
    Never folded into `evaluate`: it scores translation/comprehension quality against
    a probe set (competence.probe_set in config.yaml), not sycophancy, and reports a
    per-model verdict (Understands/Partial/Poor), not AGS/DAS/RPS/MRS/ATS/AIS.
    """
    cfg = load_config(args.config)
    if args.target_models:
        cfg.run.target_model_ids = args.target_models
    result = competence.run_competence_sweep(cfg, mock=args.mock)
    print(f"competence scores  -> {result['paths']['competence_scores']}")
    print(f"competence summary -> {result['paths']['competence_summary']}")
    for row in result["summary"]:
        if row["direction"] != "overall":
            continue
        bleu = f"{row['bleu']:.2f}" if row["bleu"] is not None else "n/a"
        chrfpp = f"{row['chrfpp']:.2f}" if row["chrfpp"] is not None else "n/a"
        print(f"  {row['model']:<24} n={row['n_items']:<3} BLEU={bleu:<8} chrF++={chrfpp:<8} {row['verdict']}")


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
    e.add_argument(
    "--source",
    choices=["tqa_style", "csqa_style", "authored"],
    help="Only evaluate items from a specific source."
)
    e.add_argument("--from-end", action="store_true",
                   help="with --limit/--limit-total, keep the LAST N items per behaviour "
                        "instead of the first N -- covers a different slice of each seed/"
                        "authored file than the default")
    e.add_argument("--target-models", nargs="*", default=None,
                   help="subset of target_models to sweep, by id or label "
                        "(e.g. --target-models Llama-3.1-8B); default: all configured targets")
    e.add_argument("--default-provider", default=None,
                   help="override run.default_provider for this run (e.g. --default-provider groq)")
    e.add_argument("--max-workers", type=int, default=0,
                   help="override run.max_workers for this run; useful for finding the "
                        "throughput ceiling without editing config.yaml")
    e.add_argument(
    "--output-dir",
    default=None,
    help="Directory where evaluation outputs will be written. "
         "Overrides run.output_dir from config.yaml."
    )
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

    m = sub.add_parser("competence", help="Nepali language competence probe (BLEU/chrF++, separate from evaluate)")
    m.add_argument("--target-models", nargs="*", default=None,
                   help="subset of target_models to probe, by id or label; default: all configured targets")
    m.add_argument("--mock", action="store_true", help="offline dry run, no API key needed")
    m.set_defaults(func=cmd_competence)

    args = ap.parse_args()
    args.func(args)
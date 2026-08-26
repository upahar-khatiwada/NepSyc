#!/usr/bin/env python3
"""Representation-analysis extraction: hidden states, attention, and answer-position
logits for open-weight target models, across sycophantic (existing dataset conditions)
and neutral (nepsyc/neutral_templates.py) prompt variants.

This is the CLI/IO driver for nepsyc/representation.py, following the same split
scripts/analyze_hidden_states.py already uses for nepsyc/hidden_states.py: pure
extraction logic lives in the module, file I/O and model/item selection live here. Only
target_models with `hf_repo_id` set in config.yaml are eligible -- the same convention
analyze_hidden_states.py uses (see its own docstring / config.yaml's comment on the field).

    python scripts/extract_representations.py --dry-run
    python scripts/extract_representations.py --model GPT-OSS-20B --behaviour agreement_bias --limit 2
    python scripts/extract_representations.py --behaviour agreement_bias delusion_acceptance --language en ne

For each (model x pair x variant x language), writes under
results/representations/ (gitignored, like the rest of results/):

  <model_label>/<behaviour>/<domain>/<language>/<pair_id>/<variant>/tensors.npz
      Per turn t, layer L: t{t}_L{L}_last_token, t{t}_L{L}_mean_pooled (float16,
      (hidden_dim,)); t{t}_L{L}_attn for configured --attn-layers (float16, head-averaged,
      (seq, seq)); t{t}_logits_next_token (float16, (vocab_size,)). See
      nepsyc/representation.py's module docstring for what last_token/mean_pooled mean.
  <model_label>/.../<pair_id>/<variant>/meta.json
      turns, replies, per-turn confidence/logit metrics (nepsyc.representation
      .confidence_logit_metrics -- null where the item has no single correct answer to
      score against, i.e. every `paired`-mode behaviour), n_layers/hidden_size/vocab_size,
      attn_layers, truncated flags.
  index.csv
      One row per (model, behaviour, domain, language, pair_id, variant, condition,
      turn_index) pointing back at its tensors.npz/meta.json -- the lightweight index the
      proposal asks for, so a dashboard or notebook can load selectively instead of
      scanning every npz. Re-running overwrites matching rows rather than duplicating,
      same convention as analyze_hidden_states.py's meta.csv.
  runs/<run_id>.json
      Reproducibility record for this invocation: cli args, per-model hf_repo_id /
      transformers+torch versions / dtype / device / tokenizer class, and the pooling
      convention description (verbatim from nepsyc/representation.py's docstring) --
      so a stored tensor's meaning never has to be guessed from code later.

--dry-run lists exactly what would be extracted (model x pair x variant x language
counts) with no model load and no torch/transformers import. --limit N keeps each
behaviour's first N items (same "per behaviour, not total" convention as run.py evaluate
--limit-per-behaviour). Resumable: a (model, pair, variant) whose tensors.npz already exists
is skipped rather than re-extracted, so killing a run and re-invoking the same command picks
up where it left off instead of reloading the model and redoing everything -- pass --force to
always re-extract regardless (needed after changing --attn-layers/--max-new-tokens/
--max-input-tokens, since those aren't part of the skip check).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.build_dataset import load  # noqa: E402
from nepsyc.config import ModelSpec, load_config  # noqa: E402
from nepsyc.neutral_templates import NO_NEUTRAL_NEEDED, load_raw_index, neutral_turns  # noqa: E402
from nepsyc.representation import confidence_logit_metrics, run_condition_full  # noqa: E402

OUT_DIR = ROOT / "results" / "representations"
INDEX_PATH = OUT_DIR / "index.csv"
RUNS_DIR = OUT_DIR / "runs"

INDEX_COLUMNS = [
    "model_label", "hf_repo_id", "behaviour", "metric", "domain", "language", "pair_id",
    "seed_id", "topic", "variant", "is_neutral_proxy", "condition", "turn_index", "n_turns",
    "n_layers", "hidden_size", "vocab_size", "attn_layers", "truncated", "device_used",
    "tensors_path", "meta_path",
    "p_correct", "p_incorrect", "logit_correct", "logit_incorrect", "logit_preference_shift",
    "correct_token", "incorrect_token",
    "prompt_preview", "generated_at",
]

POOLING_DESCRIPTION = {
    "last_token": "final-token hidden state, forward pass over the prompt only (pre-generation)",
    "mean_pooled": "mean over generated reply token positions, teacher-forced prompt+reply forward pass",
    "attn": "head-averaged attention matrix, from the teacher-forced prompt+reply forward pass",
    "next_token_logits": "full-vocab logits at the position immediately after the prompt",
}


def _select_models(cfg, wanted: Optional[List[str]]) -> List[ModelSpec]:
    eligible = [m for m in cfg.target_models if m.hf_repo_id]
    if not eligible:
        raise SystemExit(
            "No target_models in config.yaml have hf_repo_id set -- nothing to run. "
            "Add `hf_repo_id: <hf hub repo>` to at least one entry."
        )
    if not wanted:
        return eligible
    by_key = {m.label: m for m in eligible}
    by_key.update({m.id: m for m in eligible})
    selected, missing = [], []
    for w in wanted:
        (selected if w in by_key else missing).append(by_key.get(w, w))
    if missing:
        raise SystemExit(
            f"Not eligible (no hf_repo_id configured) or not a target model at all: {missing}. "
            f"Eligible: {[m.label for m in eligible]}"
        )
    return selected


def _select_items(items: List[Dict[str, Any]], behaviours: Optional[List[str]], limit: int) -> List[Dict[str, Any]]:
    if behaviours:
        items = [it for it in items if it["behaviour"] in behaviours]
    if limit <= 0:
        return items
    kept: List[Dict[str, Any]] = []
    seen_per_behaviour: Dict[str, int] = {}
    for it in items:
        b = it["behaviour"]
        seen_per_behaviour[b] = seen_per_behaviour.get(b, 0)
        if seen_per_behaviour[b] < limit:
            kept.append(it)
            seen_per_behaviour[b] += 1
    return kept


def _item_variants(item: Dict[str, Any], language: str, raw_index: Dict[str, Any]):
    """Yields (variant_name, condition_name, turns, is_neutral_proxy) for one item: every
    existing dataset condition, plus a synthetic "neutral" one where applicable (see
    nepsyc/neutral_templates.py's pairing convention -- attribution_bias's "anonymous"
    condition doubles as neutral instead of getting a separate synthetic one)."""
    for cond_name, cond in item["conditions"].items():
        is_proxy = item["behaviour"] in NO_NEUTRAL_NEEDED and cond_name == "anonymous"
        yield cond_name, cond_name, cond["turns"], is_proxy
    if item["behaviour"] not in NO_NEUTRAL_NEEDED:
        nt = neutral_turns(item, language, raw_index)
        if nt is not None:
            yield "neutral", "neutral", nt, False


def _pair_dir(model_label: str, behaviour: str, domain: str, language: str, pair_id: str, variant: str) -> Path:
    return OUT_DIR / model_label / behaviour / domain / language / pair_id / variant


def _dry_run(cfg, selected: List[ModelSpec], languages: List[str], behaviours: Optional[List[str]], limit: int) -> None:
    total = 0
    print("DRY RUN -- no model will be loaded, nothing written.\n")
    for language in languages:
        dataset_path = ROOT / "data" / f"nepsyc_{language}.csv"
        if not dataset_path.exists():
            print(f"[{language}] {dataset_path} does not exist -- run `python run.py build` first. Skipping.")
            continue
        items = _select_items(load(dataset_path), behaviours, limit)
        raw_index = load_raw_index(language)
        for spec in selected:
            n_variants = 0
            for it in items:
                n_variants += sum(1 for _ in _item_variants(it, language, raw_index))
            n_turns = sum(
                len(turns) for it in items for _, _, turns, _ in _item_variants(it, language, raw_index)
            )
            total += n_turns
            print(f"[{spec.label} / {language}] {len(items)} item(s), {n_variants} (item x variant) pair(s), "
                  f"{n_turns} turn(s) total")
            by_behaviour: Dict[str, int] = {}
            for it in items:
                by_behaviour[it["behaviour"]] = by_behaviour.get(it["behaviour"], 0) + 1
            for b, n in sorted(by_behaviour.items()):
                print(f"    {b}: {n} item(s)")
    print(f"\nTotal turns across all selected models/languages: {total}")


def run_extraction(
    selected: List[ModelSpec],
    *,
    languages: List[str],
    behaviours: Optional[List[str]] = None,
    limit: int = 0,
    items_by_language: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    attn_layers: str = "last",
    max_new_tokens: int = 120,
    max_input_tokens: int = 4096,
    device: Optional[str] = None,
    force: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Does the actual extraction for `selected` models -- the callable core behind `main()`,
    so a caller other than the CLI (the dashboard's auto-extraction, see app/dashboard.py) can
    run this in-process and get the run's metadata back, the same split pipeline.py already
    uses for run_evaluation vs. cli.cmd_evaluate.

    `items_by_language`, if given, pins extraction to an exact pre-selected item list per
    language (item dicts shaped like `nepsyc.build_dataset.load()`'s output -- e.g. the literal
    `items` pipeline.run_evaluation() just scored) instead of reloading data/nepsyc_<language>.csv
    and reselecting with `behaviours`/`limit`. This is how the dashboard keeps representation
    extraction covering exactly the prompts a sweep benchmarked rather than an independently
    chosen slice; `behaviours`/`limit` are ignored for any language present in this dict (a
    language missing from it, or the dict being None entirely, still falls back to the
    reload-and-reselect path below -- the CLI's `main()` never passes this, so it always takes
    that path).

    Unlike `main()`, one model's load/extraction failure (OOM, a checkpoint too large for this
    machine, a network error fetching the weights, ...) does not abort the remaining models --
    it's recorded in the returned `model_errors` list and the loop moves on. This matters most
    for a caller iterating over several open-weight models unattended, where a single
    known-oversized checkpoint (e.g. GPT-OSS-20B on a 16GB machine, see config.yaml's own
    comment on that entry) shouldn't take the whole run down with it. The CLI's `main()` still
    surfaces those errors (non-zero exit via the printed FAILED lines), just doesn't stop early.

    `progress`, if given, is called with a short human-readable string at coarse milestones
    (model load start, per-language item count, per-model failure) -- independent of the
    print()s below, which always fire for terminal use regardless of whether `progress` is set.

    Resumable like the sycophancy sweep's ResponseCache, but at (model, language, pair_id,
    variant) granularity rather than per-turn: before running the (expensive -- a full model
    forward pass per turn) extraction for a variant, checks whether its tensors.npz already
    exists on disk and skips straight to the next variant if so, recording it in the returned
    `resumed` list. This is what lets a killed-and-restarted extraction (e.g. the dashboard's
    auto-extract after an interrupted sweep or item spot-check) pick back up instead of
    reloading the model and redoing every item from scratch. Pass `force=True` to bypass this
    and always re-extract (e.g. after changing --attn-layers/--max-new-tokens/--max-input-tokens,
    which are not part of the skip check).
    """
    from nepsyc.representation import _resolve_attn_layers, load_model  # torch-touching imports, kept lazy

    def _tick(msg: str) -> None:
        if progress is not None:
            progress(msg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    index_df = pd.read_csv(INDEX_PATH) if INDEX_PATH.exists() else pd.DataFrame(columns=INDEX_COLUMNS)

    run_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = run_started.replace(":", "").replace("+00:00", "Z")
    run_meta: Dict[str, Any] = {
        "run_id": run_id, "started_at": run_started,
        "args": {"models": [m.label for m in selected], "languages": languages, "behaviours": behaviours,
                 "limit": limit, "attn_layers": attn_layers, "max_new_tokens": max_new_tokens,
                 "max_input_tokens": max_input_tokens, "device": device, "force": force},
        "pooling": POOLING_DESCRIPTION,
        "attn_layer_offset_note": "attn layer index i is the i-th transformer block's output "
                                   "(0-based); hidden_states/last_token/mean_pooled layer index "
                                   "is offset by +1 relative to that (index 0 = embedding layer).",
        "models": [], "skipped": [], "resumed": [], "model_errors": [],
    }

    for spec in selected:
        # attn_layers isn't known until num_hidden_layers is (needs the model config), but
        # whether ANY attention is wanted at all (spec != "none") is known from the raw CLI
        # arg alone, and attn_implementation="eager" has to be picked at load time -- sdpa
        # (the transformers default) silently returns None for attentions otherwise.
        want_attn = (attn_layers or "last").strip().lower() != "none"
        print(f"[{spec.label}] loading {spec.hf_repo_id} ...")
        _tick(f"[{spec.label}] loading {spec.hf_repo_id} ...")
        model = None
        try:
            tokenizer, model, resolved_device = load_model(
                spec.hf_repo_id, device, attn_implementation="eager" if want_attn else None,
            )
            import torch  # noqa: E402 (already loaded by load_model; safe to import here for versions/dtype)
            import transformers

            num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
            if num_hidden_layers is None:
                num_hidden_layers = len(model(**tokenizer("x", return_tensors="pt").to(resolved_device),
                                               output_hidden_states=True).hidden_states) - 1
            resolved_attn_layers = _resolve_attn_layers(attn_layers, num_hidden_layers)
            dtype = str(next(model.parameters()).dtype)

            model_meta = {
                "label": spec.label, "hf_repo_id": spec.hf_repo_id, "num_hidden_layers": num_hidden_layers,
                "attn_layers": resolved_attn_layers, "dtype": dtype, "device": resolved_device,
                "tokenizer_class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
                "transformers_version": transformers.__version__, "torch_version": torch.__version__,
            }
            run_meta["models"].append(model_meta)

            for language in languages:
                if items_by_language is not None:
                    items = items_by_language.get(language, [])
                    if not items:
                        print(f"  [{language}] no pre-selected items supplied -- skipping.")
                        continue
                else:
                    dataset_path = ROOT / "data" / f"nepsyc_{language}.csv"
                    if not dataset_path.exists():
                        print(f"  [{language}] {dataset_path} missing -- run `python run.py build` first. Skipping.")
                        continue
                    items = _select_items(load(dataset_path), behaviours, limit)
                raw_index = load_raw_index(language)
                print(f"  [{language}] {len(items)} item(s) selected")
                _tick(f"[{spec.label}] {language}: {len(items)} item(s) selected")

                for it in items:
                    pair_id, behaviour, domain = it["item_id"], it["behaviour"], it["domain"]
                    for variant, cond_name, turns, is_proxy in _item_variants(it, language, raw_index):
                        out_dir = _pair_dir(spec.label, behaviour, domain, language, pair_id, variant)
                        if not force and (out_dir / "tensors.npz").exists():
                            print(f"    {pair_id}/{variant}: already extracted -- skipping (force=True / --force to redo)")
                            run_meta["resumed"].append({"model": spec.label, "language": language,
                                                         "pair_id": pair_id, "variant": variant})
                            continue
                        try:
                            turn_reps, resolved_device = run_condition_full(
                                tokenizer, model, resolved_device, cond_name, turns,
                                max_new_tokens, resolved_attn_layers, max_input_tokens,
                            )
                        except RuntimeError as e:
                            print(f"    SKIPPED {pair_id}/{variant}: {e}", file=sys.stderr)
                            run_meta["skipped"].append({"model": spec.label, "language": language,
                                                         "pair_id": pair_id, "variant": variant, "error": str(e)})
                            continue

                        out_dir.mkdir(parents=True, exist_ok=True)
                        tensors: Dict[str, np.ndarray] = {}
                        meta_turns = []
                        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

                        for tr in turn_reps:
                            t = tr.turn_index
                            for L, vec in enumerate(tr.last_token):
                                tensors[f"t{t}_L{L}_last_token"] = vec.astype(np.float16)
                            for L, vec in enumerate(tr.mean_pooled):
                                tensors[f"t{t}_L{L}_mean_pooled"] = vec.astype(np.float16)
                            for L, mat in tr.attentions.items():
                                tensors[f"t{t}_L{L}_attn"] = mat.astype(np.float16)
                            conf = confidence_logit_metrics(tokenizer, tr.next_token_logits, it["grading"]) \
                                if tr.next_token_logits is not None else None
                            if tr.next_token_logits is not None:
                                tensors[f"t{t}_logits_next_token"] = tr.next_token_logits.astype(np.float16)

                            meta_turns.append({
                                "turn_index": t, "prompt": tr.prompt_text, "reply": tr.reply,
                                "n_layers": tr.n_layers, "hidden_size": tr.hidden_size,
                                "vocab_size": tr.vocab_size, "truncated": tr.truncated,
                                "device_used": tr.device_used, "confidence": conf,
                            })

                            index_df = index_df[~(
                                (index_df["model_label"] == spec.label) & (index_df["language"] == language)
                                & (index_df["pair_id"] == pair_id) & (index_df["variant"] == variant)
                                & (index_df["turn_index"] == t)
                            )]
                            index_df = pd.concat([index_df, pd.DataFrame([{
                                "model_label": spec.label, "hf_repo_id": spec.hf_repo_id,
                                "behaviour": behaviour, "metric": it["metric"], "domain": domain,
                                "language": language, "pair_id": pair_id, "seed_id": it["seed_id"],
                                "topic": it.get("topic"), "variant": variant, "is_neutral_proxy": is_proxy,
                                "condition": cond_name, "turn_index": t, "n_turns": len(turn_reps),
                                "n_layers": tr.n_layers, "hidden_size": tr.hidden_size,
                                "vocab_size": tr.vocab_size, "attn_layers": ",".join(map(str, resolved_attn_layers)),
                                "truncated": tr.truncated, "device_used": tr.device_used,
                                "tensors_path": str((out_dir / "tensors.npz").relative_to(ROOT).as_posix()),
                                "meta_path": str((out_dir / "meta.json").relative_to(ROOT).as_posix()),
                                "p_correct": conf["p_correct"] if conf else None,
                                "p_incorrect": conf["p_incorrect"] if conf else None,
                                "logit_correct": conf["logit_correct"] if conf else None,
                                "logit_incorrect": conf["logit_incorrect"] if conf else None,
                                "logit_preference_shift": conf["logit_preference_shift"] if conf else None,
                                "correct_token": conf["correct_token"] if conf else None,
                                "incorrect_token": conf["incorrect_token"] if conf else None,
                                "prompt_preview": tr.prompt_text[:80],
                                "generated_at": generated_at,
                            }])], ignore_index=True)

                        np.savez_compressed(out_dir / "tensors.npz", **tensors)
                        (out_dir / "meta.json").write_text(
                            json.dumps({"condition": cond_name, "is_neutral_proxy": is_proxy, "turns": meta_turns},
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        print(f"    {pair_id}/{variant}: {len(turn_reps)} turn(s), "
                              f"{len(tensors)} array(s) -> {out_dir.relative_to(ROOT)}")

                    index_df.to_csv(INDEX_PATH, index=False)
                _tick(f"[{spec.label}] {language}: done")
        except Exception as e:  # noqa: BLE001 -- one model's failure shouldn't abort the rest
            print(f"[{spec.label}] FAILED: {e}", file=sys.stderr)
            run_meta["model_errors"].append({"model": spec.label, "hf_repo_id": spec.hf_repo_id, "error": str(e)})
            _tick(f"[{spec.label}] FAILED: {e}")
        finally:
            if model is not None:
                del model
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    run_meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {INDEX_PATH.relative_to(ROOT)} and runs/{run_id}.json")
    return run_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="*", dest="models", help="Target model labels/ids (default: every "
                    "configured target model with hf_repo_id set).")
    ap.add_argument("--behaviour", nargs="*", help="Subset of the six behaviours (default: all).")
    ap.add_argument("--language", nargs="*", default=["en"], choices=["en", "ne", "ne_rom"])
    ap.add_argument("--limit", type=int, default=0, help="Keep each behaviour's first N items (0 = no limit).")
    ap.add_argument("--dry-run", action="store_true", help="List the work that would be done; extract nothing.")
    ap.add_argument("--attn-layers", default="last", help="'last', 'all', 'none', or comma-separated "
                    "0-based transformer-block indices (default: 'last').")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--max-input-tokens", type=int, default=4096, help="Left-truncate prompts longer than this.")
    ap.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    ap.add_argument("--force", action="store_true", help="Re-extract every selected item/variant "
                    "even if its tensors.npz already exists on disk. Default: skip anything "
                    "already extracted, so an interrupted run can be resumed by re-running the "
                    "same command.")
    args = ap.parse_args()

    cfg = load_config()
    selected = _select_models(cfg, args.models)

    if args.dry_run:
        _dry_run(cfg, selected, args.language, args.behaviour, args.limit)
        return

    run_extraction(
        selected, languages=args.language, behaviours=args.behaviour, limit=args.limit,
        attn_layers=args.attn_layers, max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens, device=args.device, force=args.force,
    )


if __name__ == "__main__":
    main()

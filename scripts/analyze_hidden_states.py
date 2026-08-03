#!/usr/bin/env python3
"""Offline local hidden-state extraction for one NepSyc item, across open-weight models.

Unlike `run.py evaluate`, which only ever talks to models over an OpenAI-compatible API and
never sees anything but the text reply, this script downloads a target model's real weights
via Hugging Face `transformers` and runs a forward pass locally to read its internal
per-layer hidden states -- something no API can expose. That means:
  - only target_models with `hf_repo_id` set in config.yaml are eligible (an API-only model,
    e.g. OpenAI's, has no local weights to load at all -- see config.yaml's comment on the field);
  - this is a multi-GB download and a multi-minute forward pass per model, so it always runs
    as a standalone script and never inside the Streamlit dashboard process.

    python scripts/analyze_hidden_states.py --item-id AGS-tqa001 --language en
    python scripts/analyze_hidden_states.py --behaviour delusion_acceptance --index 0 --language ne
    python scripts/analyze_hidden_states.py --item-id AGS-tqa001 --models GPT-OSS-20B

Writes, under results/hidden_states/ (gitignored along with the rest of results/):
  <language>/replies.csv
      Same columns as results/raw_responses.csv, so the dashboard's existing conversation
      renderer (app/dash_common.conversation_html) works on this file unmodified.
  <language>/vectors/<model_label>__<item_id>.csv
      One row per (condition, turn_index, layer): l2_norm, pca_x, pca_y, dim_0..dim_{H-1}.
      `prompt_type` is always "sycophantic" today -- the conditions in the NepSyc dataset
      (main, pressure, stance_pro/con, ...) are themselves the sycophancy-framed prompts
      build_dataset.py produces. The column exists so a future "normal" (unframed) condition
      can be added to build_dataset.py/TEMPLATES without a schema change here; this script does
      not compute any normal-vs-sycophantic comparison yet, since there is no normal condition
      to compare against.
  meta.csv
      One row per (model_label, language, item_id) already processed, so the dashboard can
      discover what's available without scanning every vectors CSV. Re-running overwrites the
      matching row (and files) rather than duplicating.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.build_dataset import load  # noqa: E402
from nepsyc.config import ModelSpec, load_config  # noqa: E402
from nepsyc.hidden_states import load_model, pca_trajectory, run_condition  # noqa: E402

OUT_DIR = ROOT / "results" / "hidden_states"

REPLIES_COLUMNS = ["model", "item_id", "behaviour", "condition", "turn_index", "turn", "reply", "error"]
META_COLUMNS = ["model_label", "hf_repo_id", "language", "item_id", "behaviour", "seed_id",
                "n_layers", "hidden_size", "generated_at", "vectors_path", "replies_path"]


def _select_item(items: List[Dict[str, Any]], item_id: Optional[str],
                 behaviour: Optional[str], index: int) -> Dict[str, Any]:
    if item_id:
        for it in items:
            if it["item_id"] == item_id:
                return it
        raise SystemExit(f"No item with item_id {item_id!r} in this language's dataset.")
    if behaviour:
        matches = [it for it in items if it["behaviour"] == behaviour]
        if not matches:
            raise SystemExit(f"No items for behaviour {behaviour!r}.")
        if index >= len(matches):
            raise SystemExit(f"--index {index} out of range: only {len(matches)} {behaviour!r} items.")
        return matches[index]
    raise SystemExit("Pass --item-id, or --behaviour with an optional --index.")


def _select_models(cfg, wanted: Optional[List[str]]) -> List[ModelSpec]:
    eligible = [m for m in cfg.target_models if m.hf_repo_id]
    if not eligible:
        raise SystemExit(
            "No target_models in config.yaml have hf_repo_id set -- nothing to run. "
            "Add `hf_repo_id: <hf hub repo>` to at least one entry (see the comment above "
            "target_models: in config.yaml)."
        )
    if not wanted:
        return eligible
    by_key = {m.label: m for m in eligible}
    by_key.update({m.id: m for m in eligible})
    selected, missing = [], []
    for w in wanted:
        if w in by_key:
            selected.append(by_key[w])
        else:
            missing.append(w)
    if missing:
        raise SystemExit(
            f"Not eligible (no hf_repo_id configured) or not a target model at all: {missing}. "
            f"Eligible: {[m.label for m in eligible]}"
        )
    return selected


def _read_or_empty(path: Path, columns: List[str]) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="en", choices=["en", "ne", "ne_rom"])
    ap.add_argument("--item-id", help="Exact item_id, e.g. AGS-tqa001 (from data/nepsyc_<language>.csv).")
    ap.add_argument("--behaviour", help="Pick the --index'th item of this behaviour instead of --item-id.")
    ap.add_argument("--index", type=int, default=0, help="Nth item of --behaviour (default: first, 0).")
    ap.add_argument("--models", nargs="*", help="Target model labels or ids to run (default: every "
                    "configured target model that has hf_repo_id set).")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    args = ap.parse_args()

    cfg = load_config()
    selected = _select_models(cfg, args.models)

    dataset_path = ROOT / "data" / f"nepsyc_{args.language}.csv"
    if not dataset_path.exists():
        raise SystemExit(f"{dataset_path} does not exist yet -- run `python run.py build` first.")
    items = load(dataset_path)
    item = _select_item(items, args.item_id, args.behaviour, args.index)
    item_id, behaviour = item["item_id"], item["behaviour"]
    print(f"Item: {item_id} ({behaviour}, {args.language}), "
          f"{len(item['conditions'])} condition(s): {list(item['conditions'])}")

    lang_dir = OUT_DIR / args.language
    (lang_dir / "vectors").mkdir(parents=True, exist_ok=True)
    replies_path = lang_dir / "replies.csv"
    meta_path = OUT_DIR / "meta.csv"

    replies_df = _read_or_empty(replies_path, REPLIES_COLUMNS)
    meta_df = _read_or_empty(meta_path, META_COLUMNS)

    for spec in selected:
        print(f"[{spec.label}] loading {spec.hf_repo_id} ...")
        tokenizer, model, device = load_model(spec.hf_repo_id, args.device)

        vector_rows: List[Dict[str, Any]] = []
        reply_rows: List[Dict[str, Any]] = []
        n_layers = hidden_size = None

        for cond_name, cond in item["conditions"].items():
            turns = run_condition(tokenizer, model, device, cond_name, cond["turns"], args.max_new_tokens)
            for turn in turns:
                reply_rows.append({
                    "model": spec.label, "item_id": item_id, "behaviour": behaviour,
                    "condition": turn.condition, "turn_index": turn.turn_index,
                    "turn": turn.prompt_text, "reply": turn.reply, "error": "",
                })
                pca = pca_trajectory(turn.layer_vectors)
                for layer, vec in enumerate(turn.layer_vectors):
                    hidden_size = vec.shape[0]
                    row = {
                        "condition": turn.condition, "turn_index": turn.turn_index, "layer": layer,
                        "prompt_type": "sycophantic", "l2_norm": float(np.linalg.norm(vec)),
                        "pca_x": float(pca[layer, 0]), "pca_y": float(pca[layer, 1]),
                    }
                    row.update({f"dim_{i}": float(v) for i, v in enumerate(vec)})
                    vector_rows.append(row)
                n_layers = len(turn.layer_vectors)
            print(f"  {cond_name}: {len(turns)} turn(s)")

        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        vectors_path = lang_dir / "vectors" / f"{spec.label}__{item_id}.csv"
        pd.DataFrame(vector_rows).to_csv(vectors_path, index=False)

        replies_df = replies_df[~((replies_df["model"] == spec.label) & (replies_df["item_id"] == item_id))]
        replies_df = pd.concat([replies_df, pd.DataFrame(reply_rows)], ignore_index=True)

        meta_df = meta_df[~((meta_df["model_label"] == spec.label) & (meta_df["language"] == args.language)
                            & (meta_df["item_id"] == item_id))]
        meta_df = pd.concat([meta_df, pd.DataFrame([{
            "model_label": spec.label, "hf_repo_id": spec.hf_repo_id, "language": args.language,
            "item_id": item_id, "behaviour": behaviour, "seed_id": item.get("seed_id"),
            "n_layers": n_layers, "hidden_size": hidden_size,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "vectors_path": str(vectors_path.relative_to(ROOT).as_posix()),
            "replies_path": str(replies_path.relative_to(ROOT).as_posix()),
        }])], ignore_index=True)

        print(f"[{spec.label}] wrote {vectors_path.relative_to(ROOT)} ({len(vector_rows)} rows)")

    replies_df.to_csv(replies_path, index=False)
    meta_df.to_csv(meta_path, index=False)
    print(f"Updated {replies_path.relative_to(ROOT)} and {meta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

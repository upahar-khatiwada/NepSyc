#!/usr/bin/env python3
"""Representation-comparison metrics: per-layer cosine similarity between each item's
sycophantic-condition and neutral-condition hidden states, plus dashboard-ready aggregates
and a join against the confidence/logit-preference shifts scripts/extract_representations.py
already computed per turn.

This is a pure reader of results/representations/index.csv + the tensors.npz/meta.json files
it points at -- no model load, no torch/transformers import, so it runs fast and works
offline against whatever has already been extracted. Following the module/script split
scripts/extract_representations.py already uses, the one non-trivial piece of logic
(cosine_similarity) lives in nepsyc/representation.py; this script is I/O, pairing, and
aggregation only.

Neutral-pairing convention per behaviour (mirrors nepsyc/neutral_templates.py's own
docstring -- restated here since this script is what actually resolves it row-to-row):

  agreement_bias, delusion_acceptance        main            vs neutral (both 1 turn)
  revision_under_pressure                    pressure        vs neutral (pressure has 3
                                              turns; neutral is pressure's own turn 0 text
                                              verbatim, so pressure-turn0-vs-neutral is a
                                              same-text sanity check that should sit at/near
                                              cosine 1.0 -- turns 1-2 are where the pressure
                                              signal actually shows up)
  mirroring                                  stance_pro vs neutral, stance_con vs neutral
                                              (two separate comparisons per item)
  authority_influence                        self_opinion vs neutral, authority_cue vs
                                              neutral (two separate comparisons per item)
  attribution_bias                           self_authored vs anonymous (anonymous IS the
                                              neutral proxy -- is_neutral_proxy=True in the
                                              index, no separate "neutral" variant exists)

A "pair" in the tidy table below is (model, pair_id, condition, turn_index) -- i.e. one
sycophantic-condition turn compared against its item's single neutral-condition vector,
at every layer, for both pooling variants. Multi-turn conditions (pressure; a future
condition) therefore contribute one row per turn per layer per pooling, not one row per
item -- turn_index is part of the row key precisely so the dashboard can show a pressure
item's cosine similarity moving turn 0 -> 1 -> 2 without re-deriving anything.

Usage:
    python scripts/analyze_representation_drift.py
    python scripts/analyze_representation_drift.py --index results/representations/index.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.representation import cosine_similarity  # noqa: E402

DEFAULT_INDEX = ROOT / "results" / "representations" / "index.csv"
OUT_DIR = ROOT / "data" / "representation" / "metrics"

POOLINGS = ["last_token", "mean_pooled"]

TIDY_SORT_KEYS = [
    "model", "behaviour", "domain", "language", "pair_id", "condition", "turn_index",
    "pooling", "layer",
]
TIDY_COLUMNS = TIDY_SORT_KEYS + [
    "cosine_similarity", "cosine_distance", "neutral_variant",
    "syco_p_correct", "neutral_p_correct", "confidence_shift",
    "syco_logit_preference_shift", "neutral_logit_preference_shift",
    "logit_preference_shift_delta",
]


def _load_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(
            f"{index_path} does not exist -- run scripts/extract_representations.py first "
            "(this script is a pure reader of that artifact, it extracts nothing itself)."
        )
    df = pd.read_csv(index_path)
    df["turn_index"] = df["turn_index"].astype(int)
    return df


def _resolve_neutral_variant(group: pd.DataFrame) -> Optional[str]:
    """The variant name that plays "neutral" for this (model, pair_id) group: the literal
    "neutral" condition if one was extracted, else whichever condition is flagged
    is_neutral_proxy (attribution_bias's "anonymous" -- see module docstring). None if
    neither is present, which the validation step below reports as a missing pair."""
    neutral_rows = group[group["variant"] == "neutral"]
    if not neutral_rows.empty:
        return "neutral"
    proxy_rows = group[group["is_neutral_proxy"] == True]  # noqa: E712 (pandas bool column)
    if not proxy_rows.empty:
        return proxy_rows["variant"].iloc[0]
    return None


def _load_tensors(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def build_tidy_table(index_df: pd.DataFrame) -> "tuple[pd.DataFrame, Dict[str, Any]]":
    """Returns (tidy_df, validation) -- validation collects everything the task's "flag NaNs,
    mismatched shapes, or missing pairs" ask needs, without raising, so a partial extraction
    (e.g. one behaviour still mid-run) still produces a usable table for what IS present."""
    rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {
        "missing_neutral_pairs": [],       # (model, pair_id): no neutral/proxy variant found
        "missing_tensor_files": [],        # tensors_path referenced by the index but absent on disk
        "shape_mismatches": [],            # (model, pair_id, condition, turn_index, layer, pooling)
        "nan_cosine_rows": 0,
    }

    group_keys = ["model_label", "behaviour", "domain", "language", "pair_id"]
    for (model, behaviour, domain, language, pair_id), group in index_df.groupby(group_keys):
        neutral_variant = _resolve_neutral_variant(group)
        if neutral_variant is None:
            validation["missing_neutral_pairs"].append(
                {"model": model, "behaviour": behaviour, "pair_id": pair_id}
            )
            continue

        neutral_row = group[(group["variant"] == neutral_variant) & (group["turn_index"] == 0)].iloc[0]
        neutral_tensors_path = ROOT / neutral_row["tensors_path"]
        if not neutral_tensors_path.exists():
            validation["missing_tensor_files"].append(str(neutral_row["tensors_path"]))
            continue
        neutral_tensors = _load_tensors(neutral_tensors_path)
        n_layers = int(neutral_row["n_layers"])

        syco_conditions = sorted(set(group.loc[group["variant"] != neutral_variant, "variant"]))
        for condition in syco_conditions:
            cond_rows = group[group["variant"] == condition].sort_values("turn_index")
            # tensors.npz is written once per (model, pair, variant) dir and holds every turn
            # of that condition, so load it once per condition rather than per turn_index row.
            syco_tensors_path = ROOT / cond_rows.iloc[0]["tensors_path"]
            if not syco_tensors_path.exists():
                validation["missing_tensor_files"].append(str(cond_rows.iloc[0]["tensors_path"]))
                continue
            syco_tensors = _load_tensors(syco_tensors_path)

            for _, syco_row in cond_rows.iterrows():
                t = int(syco_row["turn_index"])
                confidence_shift = None
                logit_shift_delta = None
                syco_pc = syco_row.get("p_correct")
                neutral_pc = neutral_row.get("p_correct")
                syco_lps = syco_row.get("logit_preference_shift")
                neutral_lps = neutral_row.get("logit_preference_shift")
                if pd.notna(syco_pc) and pd.notna(neutral_pc):
                    confidence_shift = float(syco_pc) - float(neutral_pc)
                if pd.notna(syco_lps) and pd.notna(neutral_lps):
                    logit_shift_delta = float(syco_lps) - float(neutral_lps)

                for pooling in POOLINGS:
                    for layer in range(n_layers):
                        syco_key = f"t{t}_L{layer}_{pooling}"
                        neutral_key = f"t0_L{layer}_{pooling}"
                        if syco_key not in syco_tensors or neutral_key not in neutral_tensors:
                            continue
                        syco_vec, neutral_vec = syco_tensors[syco_key], neutral_tensors[neutral_key]
                        if syco_vec.shape != neutral_vec.shape:
                            validation["shape_mismatches"].append(
                                {"model": model, "pair_id": pair_id, "condition": condition,
                                 "turn_index": t, "layer": layer, "pooling": pooling,
                                 "syco_shape": list(syco_vec.shape), "neutral_shape": list(neutral_vec.shape)}
                            )
                            continue
                        sim = cosine_similarity(syco_vec, neutral_vec)
                        if np.isnan(sim):
                            validation["nan_cosine_rows"] += 1
                        rows.append({
                            "model": model, "behaviour": behaviour, "domain": domain,
                            "language": language, "pair_id": pair_id, "condition": condition,
                            "turn_index": t, "pooling": pooling, "layer": layer,
                            "cosine_similarity": sim, "cosine_distance": 1.0 - sim,
                            "neutral_variant": neutral_variant,
                            "syco_p_correct": float(syco_pc) if pd.notna(syco_pc) else None,
                            "neutral_p_correct": float(neutral_pc) if pd.notna(neutral_pc) else None,
                            "confidence_shift": confidence_shift,
                            "syco_logit_preference_shift": float(syco_lps) if pd.notna(syco_lps) else None,
                            "neutral_logit_preference_shift": float(neutral_lps) if pd.notna(neutral_lps) else None,
                            "logit_preference_shift_delta": logit_shift_delta,
                        })

    tidy = pd.DataFrame(rows, columns=TIDY_COLUMNS)
    tidy = tidy.sort_values(TIDY_SORT_KEYS, kind="stable").reset_index(drop=True)
    return tidy, validation


def build_aggregates(tidy: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Dashboard-ready precomputed views, all derived from the one tidy table so there is
    exactly one source of truth for the raw per-pair numbers."""
    group_cols = ["model", "behaviour", "domain", "language", "pooling", "layer"]
    layer_agg = (
        tidy.groupby(group_cols, as_index=False)
        .agg(mean_cosine_distance=("cosine_distance", "mean"),
             mean_cosine_similarity=("cosine_similarity", "mean"),
             n_pairs=("cosine_distance", "count"))
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )

    matrices: Dict[str, pd.DataFrame] = {}
    for pooling in POOLINGS:
        sub = tidy[tidy["pooling"] == pooling]
        if sub.empty:
            continue
        matrices[pooling] = (
            sub.pivot_table(index="model", columns="layer", values="cosine_distance", aggfunc="mean")
            .sort_index(axis=1)
        )

    ranking_rows: List[Dict[str, Any]] = []
    for (model, behaviour, pooling), sub in tidy.groupby(["model", "behaviour", "pooling"]):
        per_layer = sub.groupby("layer")["cosine_distance"].mean().sort_values(ascending=False)
        for rank, (layer, dist) in enumerate(per_layer.items(), start=1):
            ranking_rows.append({
                "model": model, "behaviour": behaviour, "pooling": pooling,
                "layer": int(layer), "mean_cosine_distance": float(dist), "rank": rank,
            })
    layer_ranking = pd.DataFrame(ranking_rows).sort_values(
        ["model", "behaviour", "pooling", "rank"], kind="stable"
    ).reset_index(drop=True)

    return {"layer_agg": layer_agg, "matrices": matrices, "layer_ranking": layer_ranking}


def build_summary(tidy: pd.DataFrame, validation: Dict[str, Any]) -> Dict[str, Any]:
    per_model = {}
    for model, sub in tidy.groupby("model"):
        per_model[model] = {
            "n_rows": int(len(sub)),
            "cosine_similarity_min": float(sub["cosine_similarity"].min()),
            "cosine_similarity_max": float(sub["cosine_similarity"].max()),
            "cosine_distance_min": float(sub["cosine_distance"].min()),
            "cosine_distance_max": float(sub["cosine_distance"].max()),
            "behaviours": sorted(sub["behaviour"].unique().tolist()),
            "pair_ids": sorted(sub["pair_id"].unique().tolist()),
        }
    return {
        "n_tidy_rows": int(len(tidy)),
        "models": per_model,
        "validation": {
            "missing_neutral_pairs": validation["missing_neutral_pairs"],
            "missing_tensor_files": validation["missing_tensor_files"],
            "shape_mismatches": validation["shape_mismatches"],
            "nan_cosine_rows": validation["nan_cosine_rows"],
            "ok": (
                not validation["missing_neutral_pairs"]
                and not validation["missing_tensor_files"]
                and not validation["shape_mismatches"]
                and validation["nan_cosine_rows"] == 0
            ),
        },
    }


def run_drift_analysis(
    index_path: Path = DEFAULT_INDEX, out_dir: Path = OUT_DIR,
) -> "tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Any]]":
    """Reads index_path, writes the aggregate metric files under out_dir, and returns
    (tidy, aggregates, summary) -- the callable core behind `main()`, so a caller other than
    the CLI (the dashboard's auto-extraction, see app/dashboard.py) can regenerate these files
    in-process right after scripts/extract_representations.py runs, without shelling out.
    Raises FileNotFoundError if index_path doesn't exist, ValueError if it resolves to zero
    tidy rows -- both are caught and reported by `main()` below for CLI use.
    """
    index_df = _load_index(index_path)
    tidy, validation = build_tidy_table(index_df)
    if tidy.empty:
        raise ValueError(
            f"No (sycophantic, neutral) pairs could be resolved from {index_path} -- "
            f"validation: {json.dumps(validation, default=str)}"
        )

    aggregates = build_aggregates(tidy)
    summary = build_summary(tidy, validation)

    out_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_dir / "layer_cosine.csv", index=False)
    tidy.to_parquet(out_dir / "layer_cosine.parquet", index=False)
    aggregates["layer_agg"].to_csv(out_dir / "layer_agg.csv", index=False)
    aggregates["layer_ranking"].to_csv(out_dir / "layer_ranking.csv", index=False)
    for pooling, matrix in aggregates["matrices"].items():
        matrix.to_csv(out_dir / f"model_layer_matrix_{pooling}.csv")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tidy, aggregates, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    try:
        tidy, aggregates, summary = run_drift_analysis(args.index, args.out_dir)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e

    if not summary["validation"]["ok"]:
        print("VALIDATION FLAGS:", file=sys.stderr)
        print(json.dumps(summary["validation"], indent=2, default=str), file=sys.stderr)

    print(f"Wrote {len(tidy)} tidy rows -> {args.out_dir / 'layer_cosine.csv'} (+ .parquet)")
    print(f"Wrote layer_agg.csv, layer_ranking.csv, "
          f"model_layer_matrix_{{{','.join(aggregates['matrices'])}}}.csv, summary.json")
    print()
    print("Tidy table head:")
    print(tidy.head(10).to_string(index=False))
    print()
    one_model = tidy["model"].iloc[0]
    print(f"Per-layer aggregate for {one_model} (head):")
    print(aggregates["layer_agg"][aggregates["layer_agg"]["model"] == one_model].head(10).to_string(index=False))


if __name__ == "__main__":
    main()

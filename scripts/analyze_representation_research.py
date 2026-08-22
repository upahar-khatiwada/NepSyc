#!/usr/bin/env python3
"""Optional research panels precompute: PCA projection, sycophancy-direction (mean-diff)
vectors, linear CKA, revision-under-pressure drift trajectories, and tokenizer-fertility-
vs-drift correlation -- all derived from results/representations/index.csv +
tensors.npz/meta.json, the same artifacts scripts/analyze_representation_drift.py already
reads to compute per-layer cosine similarity.

This is a SEPARATE script, not an edit to analyze_representation_drift.py -- the same
"separate, richer, not an edit to it" precedent nepsyc/representation.py's own module
docstring already sets relative to nepsyc/hidden_states.py. analyze_representation_drift.py's
five output files (layer_cosine.csv, layer_agg.csv, layer_ranking.csv,
model_layer_matrix_*.csv, summary.json) are a documented dependency of the dashboard's core
"Representational learning" section and must not change shape or meaning. This script writes
its own, additional `research_*` files into the same data/representation/metrics/ directory
("next to" Prompt 3's metrics, not a new location) -- read only by the dashboard's optional
research panels, never by the core section.

SAMPLE SIZES ARE SMALL BY CONSTRUCTION. As of the last extraction run: 1 model
(Qwen2.5-1.5B), 11 item-pairs spread across 6 behaviours (attribution_bias has exactly 1).
Every metric below is still computed and written regardless of n -- nothing here silently
drops a low-n result -- but n is always carried alongside the number so a downstream reader
(the dashboard) can flag it instead of presenting it as settled. See each function's
docstring for the specific caveat.

Outputs (data/representation/metrics/):
  research_pca_layers.csv          one row per (model, pooling, layer, point) -- 2D PCA
                                    coordinates, fit per (model, pooling, layer) over every
                                    sycophantic-condition vector plus one neutral vector per
                                    item at that layer, variant-labelled for colouring.
  research_directions.csv          one row per (model, behaviour-or-"__all__", pooling,
                                    layer) -- ||mean_pairs(syco_vec - neutral_vec)|| (the
                                    sycophancy DIRECTION's norm; the direction itself is a
                                    derived difference-of-means, not a trained weight).
  research_directions.npz          the direction vectors themselves, one array per
                                    "<model>__<behaviour>__<pooling>__L<layer>" key --
                                    kept out of the CSV since hidden_dim (1536) columns
                                    would make it unreadable, but preserved for reuse.
  research_direction_stability.csv one row per (model, pooling, layer, behaviour_a,
                                    behaviour_b) -- cosine similarity between two
                                    behaviours' direction vectors at that layer, i.e. how
                                    much a "sycophancy direction" derived from one behaviour
                                    generalizes to another.
  research_cka.csv                 one row per (model, pooling, layer, scope) -- linear CKA
                                    between the sycophantic- and neutral-side representation
                                    geometries, scope="__all__" (every matched pair pooled)
                                    or a single behaviour name (only when it has >=3 pairs).
  research_rup_drift.csv           one row per (model, pair_id, pooling, layer, turn_index)
                                    -- revision_under_pressure only: step-wise cosine
                                    similarity/distance between consecutive pressure turns'
                                    OWN hidden states (not vs. neutral), and the cumulative
                                    sum of that step distance across turns 0->1->2, joined
                                    against the existing cosine-distance-from-neutral for
                                    context.
  research_fertility.csv           one row per (model, pooling, layer) -- Spearman rho and
                                    linear R^2 between each row's prompt tokenizer fertility
                                    (tokens per word) and its cosine distance from neutral,
                                    correlated across every matched item/condition/turn at
                                    that layer.

Usage:
    python scripts/analyze_representation_research.py
    python scripts/analyze_representation_research.py --index results/representations/index.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.representation import cosine_similarity, linear_cka  # noqa: E402

DEFAULT_INDEX = ROOT / "results" / "representations" / "index.csv"
OUT_DIR = ROOT / "data" / "representation" / "metrics"

POOLINGS = ["last_token", "mean_pooled"]
MIN_CKA_N = 3          # below this, a per-behaviour CKA number is noise, not a sketch
MIN_PCA_N = 4          # below this, 2D PCA has more components than useful structure


def _load_index(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        raise SystemExit(
            f"{index_path} does not exist -- run scripts/extract_representations.py first "
            "(this script is a pure reader of that artifact, it extracts nothing itself)."
        )
    df = pd.read_csv(index_path)
    df["turn_index"] = df["turn_index"].astype(int)
    return df


def _resolve_neutral_variant(group: pd.DataFrame) -> Optional[str]:
    """Same resolution scripts/analyze_representation_drift.py uses: the literal "neutral"
    variant if one was extracted, else whichever condition is flagged is_neutral_proxy
    (attribution_bias's "anonymous")."""
    neutral_rows = group[group["variant"] == "neutral"]
    if not neutral_rows.empty:
        return "neutral"
    proxy_rows = group[group["is_neutral_proxy"] == True]  # noqa: E712 (pandas bool column)
    if not proxy_rows.empty:
        return proxy_rows["variant"].iloc[0]
    return None


def _load_tensors(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def _load_meta(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def collect_matched_pairs(index_df: pd.DataFrame) -> "tuple[Dict, Dict, Dict]":
    """One pass over every (model, behaviour, domain, language, pair_id) group, loading
    each group's tensors exactly once, building three structures every panel below reuses
    rather than re-reading tensors.npz per panel:

      pca_points   (model, pooling, layer) -> list of {vector, variant, behaviour, pair_id,
                    condition, turn_index} -- the neutral vector appears once per pair_id
                    (deduped), every sycophantic condition/turn appears once each.
      matched      (model, behaviour, pooling, layer) -> list of (syco_vec, neutral_vec,
                    pair_id, condition, turn_index) -- paired samples for direction/CKA.
      fertility_meta  (model, pair_id, condition, turn_index) -> prompt text, for the
                    tokenizer-fertility panel (kept separate since it needs no vectors).
    """
    pca_points: Dict[tuple, List[dict]] = {}
    matched: Dict[tuple, List[tuple]] = {}
    fertility_meta: Dict[tuple, str] = {}

    group_keys = ["model_label", "behaviour", "domain", "language", "pair_id"]
    for (model, behaviour, domain, language, pair_id), group in index_df.groupby(group_keys):
        neutral_variant = _resolve_neutral_variant(group)
        if neutral_variant is None:
            continue
        neutral_row = group[(group["variant"] == neutral_variant) & (group["turn_index"] == 0)]
        if neutral_row.empty:
            continue
        neutral_row = neutral_row.iloc[0]
        neutral_tensors = _load_tensors(ROOT / neutral_row["tensors_path"])
        if neutral_tensors is None:
            continue
        n_layers = int(neutral_row["n_layers"])

        neutral_meta = _load_meta(ROOT / neutral_row["meta_path"])
        if neutral_meta and neutral_meta.get("turns"):
            fertility_meta[(model, pair_id, neutral_variant, 0)] = neutral_meta["turns"][0].get("prompt", "")

        syco_conditions = sorted(set(group.loc[group["variant"] != neutral_variant, "variant"]))
        for condition in syco_conditions:
            cond_rows = group[group["variant"] == condition].sort_values("turn_index")
            syco_tensors = _load_tensors(ROOT / cond_rows.iloc[0]["tensors_path"])
            if syco_tensors is None:
                continue
            syco_meta = _load_meta(ROOT / cond_rows.iloc[0]["meta_path"])
            meta_by_turn = {t["turn_index"]: t.get("prompt", "") for t in (syco_meta or {}).get("turns", [])}

            for _, row in cond_rows.iterrows():
                t = int(row["turn_index"])
                fertility_meta[(model, pair_id, condition, t)] = meta_by_turn.get(t, "")

                for pooling in POOLINGS:
                    for layer in range(n_layers):
                        syco_key, neutral_key = f"t{t}_L{layer}_{pooling}", f"t0_L{layer}_{pooling}"
                        if syco_key not in syco_tensors or neutral_key not in neutral_tensors:
                            continue
                        syco_vec, neutral_vec = syco_tensors[syco_key], neutral_tensors[neutral_key]
                        if syco_vec.shape != neutral_vec.shape:
                            continue

                        pca_key = (model, pooling, layer)
                        pts = pca_points.setdefault(pca_key, [])
                        pts.append({"vector": syco_vec, "variant": "sycophantic", "behaviour": behaviour,
                                    "domain": domain, "language": language, "pair_id": pair_id,
                                    "condition": condition, "turn_index": t})
                        if not any(p["pair_id"] == pair_id and p["variant"] == "neutral" for p in pts):
                            pts.append({"vector": neutral_vec, "variant": "neutral", "behaviour": behaviour,
                                        "domain": domain, "language": language, "pair_id": pair_id,
                                        "condition": neutral_variant, "turn_index": 0})

                        matched.setdefault((model, behaviour, pooling, layer), []).append(
                            (syco_vec, neutral_vec, pair_id, condition, t)
                        )

    return pca_points, matched, fertility_meta


# ---------------------------------------------------------------------------
# Panel 1: PCA projection at a chosen layer, coloured by variant
# ---------------------------------------------------------------------------

def build_pca_table(pca_points: Dict[tuple, List[dict]]) -> pd.DataFrame:
    """2D PCA fit PER (model, pooling, layer), over every point collected for that layer
    (every sycophantic condition/turn's vector, plus one neutral vector per item). This is
    a per-layer fit, not a shared cross-layer basis -- component 1/2 at layer 5 are not the
    same axes as at layer 20, matching nepsyc/hidden_states.pca_trajectory's own per-turn
    (not shared) convention. Skips (model, pooling, layer) combinations with fewer than
    MIN_PCA_N points -- 2D PCA on 2-3 points is a rotation of two/three dots, not structure."""
    rows: List[Dict[str, Any]] = []
    for (model, pooling, layer), pts in pca_points.items():
        if len(pts) < MIN_PCA_N:
            continue
        X = np.stack([p["vector"].astype(np.float64) for p in pts])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_
        for p, (x, y) in zip(pts, coords):
            rows.append({
                "model": model, "pooling": pooling, "layer": layer, "behaviour": p["behaviour"],
                "domain": p["domain"], "language": p["language"], "pair_id": p["pair_id"],
                "condition": p["condition"], "turn_index": p["turn_index"], "variant": p["variant"],
                "pca_x": float(x), "pca_y": float(y),
                "explained_variance_ratio_1": float(evr[0]), "explained_variance_ratio_2": float(evr[1]),
                "n_points": len(pts),
            })
    return pd.DataFrame(rows).sort_values(
        ["model", "pooling", "layer", "variant", "pair_id"], kind="stable"
    ).reset_index(drop=True) if rows else pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel 2: sycophancy direction (difference-in-means) + cross-behaviour stability
# ---------------------------------------------------------------------------

def build_direction_tables(matched: Dict[tuple, List[tuple]]) -> "tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]":
    """DIRECTION = mean over matched (item, condition, turn) pairs of (syco_vec -
    neutral_vec) -- a paired difference-of-means, the same construction representation-
    engineering steering vectors use (e.g. Rimsky et al.'s contrastive activation
    addition), NOT a trained probe weight. Computed per (model, behaviour, pooling, layer)
    and pooled across all behaviours as behaviour="__all__". norm(direction) is the
    headline number in research_directions.csv; the vectors themselves go to
    research_directions.npz since hidden_dim columns don't fit a CSV usefully.

    Cross-behaviour cosine stability: cosine(direction_b1, direction_b2) at the same
    (model, pooling, layer) for every pair of behaviours that both have >=1 matched pair --
    n=1 behaviours (attribution_bias) produce a direction from a single item, so their row
    is included but should be read as "this one item's own syco-neutral difference", not a
    behaviour-general direction."""
    direction_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    vectors: Dict[str, np.ndarray] = {}

    # group matched by (model, pooling, layer) -> {behaviour: [diffs]}, plus "__all__"
    by_mpl: Dict[tuple, Dict[str, List[np.ndarray]]] = {}
    for (model, behaviour, pooling, layer), pairs in matched.items():
        diffs = [s.astype(np.float64) - n.astype(np.float64) for s, n, *_ in pairs]
        key = (model, pooling, layer)
        by_mpl.setdefault(key, {}).setdefault(behaviour, []).extend(diffs)
        by_mpl[key].setdefault("__all__", []).extend(diffs)

    for (model, pooling, layer), by_behaviour in by_mpl.items():
        directions: Dict[str, np.ndarray] = {}
        for behaviour, diffs in by_behaviour.items():
            direction = np.mean(np.stack(diffs), axis=0)
            directions[behaviour] = direction
            vectors[f"{model}__{behaviour}__{pooling}__L{layer}"] = direction.astype(np.float32)
            direction_rows.append({
                "model": model, "behaviour": behaviour, "pooling": pooling, "layer": layer,
                "n_pairs": len(diffs), "direction_norm": float(np.linalg.norm(direction)),
            })

        real_behaviours = sorted(b for b in directions if b != "__all__")
        for b1, b2 in itertools.combinations(real_behaviours, 2):
            sim = cosine_similarity(directions[b1], directions[b2])
            n1 = len(by_behaviour[b1])
            n2 = len(by_behaviour[b2])
            stability_rows.append({
                "model": model, "pooling": pooling, "layer": layer,
                "behaviour_a": b1, "behaviour_b": b2, "cosine": sim,
                "n_pairs_a": n1, "n_pairs_b": n2, "low_n": (n1 < MIN_CKA_N or n2 < MIN_CKA_N),
            })

    direction_df = pd.DataFrame(direction_rows).sort_values(
        ["model", "behaviour", "pooling", "layer"], kind="stable"
    ).reset_index(drop=True) if direction_rows else pd.DataFrame(direction_rows)
    stability_df = pd.DataFrame(stability_rows).sort_values(
        ["model", "pooling", "layer", "behaviour_a", "behaviour_b"], kind="stable"
    ).reset_index(drop=True) if stability_rows else pd.DataFrame(stability_rows)
    return direction_df, stability_df, vectors


# ---------------------------------------------------------------------------
# Panel 3: linear CKA per layer, syco vs neutral
# ---------------------------------------------------------------------------

def build_cka_table(matched: Dict[tuple, List[tuple]]) -> pd.DataFrame:
    """Linear CKA (nepsyc.representation.linear_cka) between the sycophantic-side and
    neutral-side representation matrices at each (model, pooling, layer), scope="__all__"
    (every matched pair across every behaviour pooled into one batch -- the only scope with
    enough n to be more than a coin flip) plus a per-behaviour breakdown, included ONLY when
    that behaviour has >= MIN_CKA_N matched pairs at that layer (attribution_bias's n=1 is
    dropped rather than reported as a number that looks like the others but means much
    less)."""
    rows: List[Dict[str, Any]] = []
    by_mpl: Dict[tuple, Dict[str, List[tuple]]] = {}
    for (model, behaviour, pooling, layer), pairs in matched.items():
        key = (model, pooling, layer)
        by_mpl.setdefault(key, {}).setdefault(behaviour, []).extend(pairs)
        by_mpl[key].setdefault("__all__", []).extend(pairs)

    for (model, pooling, layer), by_behaviour in by_mpl.items():
        for scope, pairs in by_behaviour.items():
            n = len(pairs)
            if scope != "__all__" and n < MIN_CKA_N:
                continue
            X = np.stack([p[0].astype(np.float64) for p in pairs])
            Y = np.stack([p[1].astype(np.float64) for p in pairs])
            rows.append({
                "model": model, "pooling": pooling, "layer": layer, "scope": scope,
                "n_samples": n, "cka": linear_cka(X, Y),
            })
    return pd.DataFrame(rows).sort_values(
        ["model", "pooling", "layer", "scope"], kind="stable"
    ).reset_index(drop=True) if rows else pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel 4: revision-under-pressure drift trajectory
# ---------------------------------------------------------------------------

def build_rup_drift_table(index_df: pd.DataFrame, layer_cosine_path: Path) -> pd.DataFrame:
    """revision_under_pressure only: the "pressure" condition's own hidden states across its
    3 turns, compared turn-to-turn (t vs t-1) rather than each turn vs. the neutral anchor --
    this is a genuinely different signal from the core section's cosine-vs-neutral chart: it
    asks "how far has the model's internal state moved between successive pressure turns",
    independent of where it started. cumulative_drift is the running sum of (1 -
    step_cosine_similarity) across turns 0->1->2, i.e. total path length in cosine-distance
    terms, not just net displacement.

    cosine_distance_from_neutral is joined in from research's sibling
    scripts/analyze_representation_drift.py output (layer_cosine.csv) purely for context --
    no new tensor loads needed for that column, it already exists.

    Only 2 revision_under_pressure items exist as of the last extraction run -- read any
    single trajectory as one model's behaviour on one item, not a population statistic."""
    rup = index_df[(index_df["behaviour"] == "revision_under_pressure") & (index_df["condition"] == "pressure")]
    rows: List[Dict[str, Any]] = []
    for (model, pair_id, domain, language), group in rup.groupby(["model_label", "pair_id", "domain", "language"]):
        group = group.sort_values("turn_index")
        tensors_path = ROOT / group.iloc[0]["tensors_path"]
        tensors = _load_tensors(tensors_path)
        if tensors is None:
            continue
        n_layers = int(group.iloc[0]["n_layers"])
        turns = sorted(group["turn_index"].unique().tolist())

        for pooling in POOLINGS:
            for layer in range(n_layers):
                cumulative = 0.0
                for i, t in enumerate(turns):
                    key = f"t{t}_L{layer}_{pooling}"
                    if key not in tensors:
                        continue
                    if i == 0:
                        step_sim, step_dist = None, None
                    else:
                        prev_key = f"t{turns[i - 1]}_L{layer}_{pooling}"
                        if prev_key not in tensors:
                            step_sim, step_dist = None, None
                        else:
                            step_sim = cosine_similarity(tensors[key], tensors[prev_key])
                            step_dist = 1.0 - step_sim if step_sim == step_sim else float("nan")
                            if step_dist == step_dist:
                                cumulative += step_dist
                    rows.append({
                        "model": model, "pair_id": pair_id, "domain": domain, "language": language,
                        "pooling": pooling, "layer": layer, "turn_index": t,
                        "step_cosine_similarity": step_sim, "step_cosine_distance": step_dist,
                        "cumulative_drift": cumulative,
                    })

    rup_df = pd.DataFrame(rows)
    if rup_df.empty or not layer_cosine_path.exists():
        return rup_df.sort_values(["model", "pair_id", "pooling", "layer", "turn_index"], kind="stable").reset_index(drop=True) if not rup_df.empty else rup_df

    layer_cosine = pd.read_csv(layer_cosine_path)
    neutral_ctx = layer_cosine[layer_cosine["condition"] == "pressure"][
        ["model", "pair_id", "pooling", "layer", "turn_index", "cosine_distance"]
    ].rename(columns={"cosine_distance": "cosine_distance_from_neutral"})
    rup_df = rup_df.merge(neutral_ctx, on=["model", "pair_id", "pooling", "layer", "turn_index"], how="left")
    return rup_df.sort_values(["model", "pair_id", "pooling", "layer", "turn_index"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Panel 5: tokenizer fertility vs. per-layer drift
# ---------------------------------------------------------------------------

def _fertility(tokenizer, text: str) -> Optional[float]:
    text = (text or "").strip()
    n_words = len(text.split())
    if n_words == 0:
        return None
    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return n_tokens / n_words


def build_fertility_table(
    index_df: pd.DataFrame, fertility_meta: Dict[tuple, str], layer_cosine_path: Path,
) -> pd.DataFrame:
    """Tokenizer fertility (tokens per whitespace word) of each prompt, correlated against
    that same (model, pair_id, condition, turn_index) row's cosine distance from its neutral
    twin -- per (model, pooling, layer), across every matched item/condition/turn available
    at that layer. Fertility itself doesn't vary by layer (it's a property of the tokenizer
    + prompt text); what's being asked is whether layers differ in how strongly a prompt's
    subword fragmentation predicts how far its representation drifts from neutral.

    Only 1 model has been extracted so far, so this can't yet show a cross-model pattern --
    it reports the correlation for whichever model(s) ARE present, one row per (model,
    pooling, layer), and the caller should read "per model" literally: n=1 model today.

    n per layer is the item/condition/turn count available (currently ~20-40) -- Spearman
    rho and R^2 are computed and returned regardless of n, but this script does not decide a
    significance threshold; that judgment (and the "few authored items, treat with caution"
    caveat) belongs to the caller/dashboard caption, not silently baked in here."""
    if not layer_cosine_path.exists():
        return pd.DataFrame()
    layer_cosine = pd.read_csv(layer_cosine_path)

    hf_repo_by_model = dict(zip(index_df["model_label"], index_df["hf_repo_id"]))
    fertility_rows = []
    for model in sorted(hf_repo_by_model):
        hf_repo_id = hf_repo_by_model[model]
        try:
            from transformers import AutoTokenizer  # lazy: only this panel touches transformers
            tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
        except Exception as e:  # noqa: BLE001 -- offline/uncached repo, skip this model's fertility only
            print(f"  [fertility] could not load tokenizer for {model} ({hf_repo_id}): {e}", file=sys.stderr)
            continue
        for (m, pair_id, condition, turn_index), text in fertility_meta.items():
            if m != model:
                continue
            fert = _fertility(tokenizer, text)
            if fert is not None:
                fertility_rows.append({
                    "model": model, "pair_id": pair_id, "condition": condition,
                    "turn_index": turn_index, "fertility": fert,
                })
    if not fertility_rows:
        return pd.DataFrame()
    fert_df = pd.DataFrame(fertility_rows)

    joined = layer_cosine.merge(
        fert_df, on=["model", "pair_id", "condition", "turn_index"], how="inner",
    )
    rows: List[Dict[str, Any]] = []
    for (model, pooling, layer), sub in joined.groupby(["model", "pooling", "layer"]):
        sub = sub.dropna(subset=["fertility", "cosine_distance"])
        n = len(sub)
        if n < 3:
            rows.append({"model": model, "pooling": pooling, "layer": layer, "n_points": n,
                         "spearman_rho": None, "spearman_p": None, "r_squared": None})
            continue
        rho, p = stats.spearmanr(sub["fertility"], sub["cosine_distance"])
        lin = stats.linregress(sub["fertility"], sub["cosine_distance"])
        rows.append({
            "model": model, "pooling": pooling, "layer": layer, "n_points": n,
            "spearman_rho": float(rho), "spearman_p": float(p), "r_squared": float(lin.rvalue ** 2),
        })
    return pd.DataFrame(rows).sort_values(["model", "pooling", "layer"], kind="stable").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-fertility", action="store_true",
                    help="Skip the tokenizer-fertility panel (it loads a tokenizer per model; "
                         "everything else here is pure numpy/pandas over already-extracted tensors).")
    args = ap.parse_args()

    index_df = _load_index(args.index)
    layer_cosine_path = args.out_dir / "layer_cosine.csv"
    if not layer_cosine_path.exists():
        raise SystemExit(
            f"{layer_cosine_path} does not exist -- run scripts/analyze_representation_drift.py "
            "first (this script reuses its cosine-vs-neutral output for context/correlation)."
        )

    print("Collecting matched sycophantic/neutral pairs from tensors.npz ...")
    pca_points, matched, fertility_meta = collect_matched_pairs(index_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pca_df = build_pca_table(pca_points)
    pca_df.to_csv(args.out_dir / "research_pca_layers.csv", index=False)
    n_fits = pca_df.drop_duplicates(["model", "pooling", "layer"]).shape[0] if not pca_df.empty else 0
    print(f"  research_pca_layers.csv: {len(pca_df)} rows ({n_fits} (model,pooling,layer) fits with n>={MIN_PCA_N})")

    direction_df, stability_df, direction_vectors = build_direction_tables(matched)
    direction_df.to_csv(args.out_dir / "research_directions.csv", index=False)
    stability_df.to_csv(args.out_dir / "research_direction_stability.csv", index=False)
    np.savez_compressed(args.out_dir / "research_directions.npz", **direction_vectors)
    print(f"  research_directions.csv: {len(direction_df)} rows, "
          f"research_direction_stability.csv: {len(stability_df)} rows")

    cka_df = build_cka_table(matched)
    cka_df.to_csv(args.out_dir / "research_cka.csv", index=False)
    print(f"  research_cka.csv: {len(cka_df)} rows")

    rup_df = build_rup_drift_table(index_df, layer_cosine_path)
    rup_df.to_csv(args.out_dir / "research_rup_drift.csv", index=False)
    print(f"  research_rup_drift.csv: {len(rup_df)} rows "
          f"({rup_df['pair_id'].nunique() if not rup_df.empty else 0} revision_under_pressure item(s))")

    if args.skip_fertility:
        print("  research_fertility.csv: skipped (--skip-fertility)")
    else:
        fert_df = build_fertility_table(index_df, fertility_meta, layer_cosine_path)
        fert_df.to_csv(args.out_dir / "research_fertility.csv", index=False)
        print(f"  research_fertility.csv: {len(fert_df)} rows")

    print(f"\nWrote research_*.csv/.npz to {args.out_dir}")


if __name__ == "__main__":
    main()

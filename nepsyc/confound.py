"""E7: Baseline-Accuracy Confound Check.

Tests whether apparent multilingual differences in sycophancy scores are partly explained
by language competence, using data the evaluation pipeline already collects -- no new
model inference. The key fact this leans on: RPS's turn 1 is a neutral, unpressured
question ("{question}... Please answer...", no leading claim), scored via the same
`eligible` flag `metrics.score_item()` already sets for every revision_under_pressure item.
That flag is a per-item, per-model, per-language competence signal that happens to already
exist for a different reason (RPS excludes turn-1-wrong items from capitulation scoring,
since "capitulation" is undefined for a model that never knew the answer).

AGS-mcq and RPS-mcq items share `seed_id` (same underlying `mcq` seed list in
build_dataset.py, verified against the actual dataset, not assumed -- see the accompanying
analysis writeup) and both come from `source == "csqa_style"`. That's what lets H3 borrow
RPS's neutral-turn competence signal to look for a confound in AGS specifically.

Three analyses:
    H1  Pooled RPS `eligible` rate per language, with bootstrap CI -- is there a
        competence gap at all?
    H2  RPS flip_rate / stable_correct_rate on the full eligible pool vs. a
        triple-language-eligible matched subset, per model -- does comparing the SAME
        items across languages (rather than each language's own eligible pool, which may
        be a different, easier subset) change the headline numbers?
    H3  2x2 (RPS-neutral-eligible x AGS-agreed-with-false-claim) contingency table on the
        seed_id-matched csqa_style subset, Fisher's exact test + odds ratio + Woolf CI.

All three reuse `metrics.aggregate()` / `metrics.bootstrap_ci()` rather than re-deriving
their formulas, so results stay correct automatically if those definitions change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from .metrics import aggregate, bootstrap_ci

LANGUAGES = ["en", "ne", "ne_rom"]

# Columns score_item() puts in detail_json that H1/H2/H3 need pulled back out as real
# DataFrame columns. Deliberately narrow -- this is not a general detail_json unpacker,
# just what this analysis touches.
_DETAIL_COLUMNS = ["eligible", "flip", "stable_correct", "recovery",
                   "hard_correct", "hard_agreed_with_user"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_item_scores(path: Path) -> pd.DataFrame:
    """Read one language's item_scores.csv and flatten the detail_json fields this
    analysis needs into real columns, the same way app/dashboard.py's Prompt Inspector
    already parses this file (json.loads per row, malformed/missing -> skipped, never
    raises) -- not a new convention.
    """
    df = pd.read_csv(path)
    for col in _DETAIL_COLUMNS:
        df[col] = None
    if "detail_json" not in df.columns:
        raise ValueError(f"{path} has no detail_json column -- not a NepSyc item_scores.csv?")
    for i, raw in df["detail_json"].items():
        if not isinstance(raw, str) or not raw:
            continue
        try:
            detail = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for col in _DETAIL_COLUMNS:
            if col in detail:
                df.at[i, col] = detail[col]
    return df


def load_all_languages(paths: Dict[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load all three languages' item_scores.csv, failing loudly and specifically if any
    are missing -- deliberately not a silent skip, because `cli.py`'s `evaluate` command
    has no --output-dir flag and overwrites results/item_scores.csv on every run, so
    "only 2 of 3 languages present" is a likely real mistake, not an edge case to shrug off.
    """
    missing = [lang for lang in LANGUAGES if lang not in paths or not Path(paths[lang]).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing item_scores.csv for: {missing}. E7 needs all three languages' "
            f"item_scores.csv at once, but `python run.py evaluate` has no --output-dir "
            f"flag and overwrites results/item_scores.csv on every run. Either run each "
            f"language through app/dashboard.py's multi-language mode (writes "
            f"results/<language>/item_scores.csv automatically), or after each CLI run "
            f"move results/item_scores.csv to results/<language>/item_scores.csv by hand "
            f"before running the next language."
        )
    return {lang: load_item_scores(Path(paths[lang])) for lang in LANGUAGES}


# ---------------------------------------------------------------------------
# H1 -- pooled baseline competence per language
# ---------------------------------------------------------------------------

def h1_baseline_competence(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RPS `eligible` rate pooled across every model and item in each language, with a
    bootstrap 95% CI (metrics.bootstrap_ci, unmodified). This is deliberately NOT split
    by model -- with only a few configured target models, a per-model breakdown here
    would be underpowered noise; see H2 for the per-model view, which only needs
    within-model comparisons (full vs. matched), not across-model ones.
    """
    rows = []
    for lang in LANGUAGES:
        df = dfs[lang]
        rps = df[df["behaviour"] == "revision_under_pressure"]
        flags = [bool(x) for x in rps["eligible"].dropna()]
        n = len(flags)
        rate = (sum(flags) / n) if n else float("nan")
        if n > 1:
            lo, hi = bootstrap_ci([float(f) for f in flags])
        else:
            lo, hi = float("nan"), float("nan")
        rows.append({"language": lang, "eligible_rate": rate, "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H2 -- matched vs. unmatched RPS subset, per model
# ---------------------------------------------------------------------------

def _score_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Reconstruct the flat per-item dict shape metrics.score_item() originally produced,
    from a DataFrame slice -- this is what lets us call metrics.aggregate() on an
    arbitrary subset (e.g. the triple-language-matched items) and get flip_rate /
    stable_correct_rate computed by the SAME code as everywhere else in the project,
    rather than re-deriving the formula here and risking it drifting out of sync.

    `metrics.aggregate()` filters on `is not None` (e.g. `r.get("score") is not None`),
    which is correct for the in-memory dicts score_item() actually produces (a missing
    score really is Python None there) but NOT for a value round-tripped through
    pandas.read_csv: a missing numeric cell becomes float NaN, and `NaN is not None`
    is True -- so without this normalization, unscored items would silently leak into
    aggregate()'s mean/stdev/flip_rate and either corrupt the numbers or crash
    (statistics.pstdev cannot handle NaN). Confirmed by triggering the crash against
    this exact reconstruction before adding the fix, not assumed.
    """
    def _clean(v: Any) -> Any:
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, np.generic):  # numpy scalar (float64/int64/bool_) -> native Python
            return v.item()
        return v

    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def h2_matched_subset(dfs: Dict[str, pd.DataFrame], min_matched_n: int = 5) -> pd.DataFrame:
    """Per model, per language: RPS flip_rate/stable_correct_rate on (a) the full set of
    items eligible in that language alone, vs. (b) the subset of items ALSO eligible for
    that same model in both other languages. (a) is what a naive per-language comparison
    reports today; (b) controls for the possibility that each language's eligible pool is
    a different, differently-easy subset of items, which would make a raw (a)-vs-(a)
    cross-language comparison partly a sampling artifact rather than a behavioural one.

    Rows where the matched subset has fewer than `min_matched_n` items get NaN for the
    matched-subset statistics and `insufficient_overlap = True`, rather than a number
    computed on (e.g.) 2 items -- a technically-real but meaningless statistic is worse
    than a clearly-flagged missing one.
    """
    rps = {lang: dfs[lang][dfs[lang]["behaviour"] == "revision_under_pressure"] for lang in LANGUAGES}
    models = set(rps["en"]["model"].dropna())
    for lang in LANGUAGES[1:]:
        models &= set(rps[lang]["model"].dropna())
    models = sorted(models)
    if not models:
        raise ValueError("No model appears in all three languages' RPS results -- nothing to compare.")

    rows = []
    for model in models:
        elig_seed_ids = {}
        per_lang_df = {}
        for lang in LANGUAGES:
            sub = rps[lang][rps[lang]["model"] == model]
            per_lang_df[lang] = sub
            elig_seed_ids[lang] = set(sub.loc[sub["eligible"] == True, "seed_id"])  # noqa: E712
        matched_seed_ids = elig_seed_ids["en"] & elig_seed_ids["ne"] & elig_seed_ids["ne_rom"]
        insufficient = len(matched_seed_ids) < min_matched_n

        for lang in LANGUAGES:
            sub = per_lang_df[lang]
            full_agg = aggregate(_score_rows(sub)).get(f"{model}||revision_under_pressure", {})
            full_flip = full_agg.get("flip_rate")
            full_stable = full_agg.get("stable_correct_rate")
            full_n = full_agg.get("n_eligible")

            if insufficient:
                matched_flip = matched_stable = float("nan")
            else:
                matched_sub = sub[sub["seed_id"].isin(matched_seed_ids)]
                matched_agg = aggregate(_score_rows(matched_sub)).get(
                    f"{model}||revision_under_pressure", {})
                matched_flip = matched_agg.get("flip_rate")
                matched_stable = matched_agg.get("stable_correct_rate")

            rows.append({
                "model": model, "language": lang,
                "flip_rate_unmatched": full_flip, "flip_rate_matched": matched_flip,
                "flip_rate_abs_diff": (None if matched_flip is None or full_flip is None
                                       else matched_flip - full_flip),
                "flip_rate_rel_diff_pct": (
                    None if matched_flip is None or full_flip is None or full_flip == 0
                    else 100.0 * (matched_flip - full_flip) / full_flip),
                "stable_correct_rate_unmatched": full_stable,
                "stable_correct_rate_matched": matched_stable,
                "n_eligible_unmatched": full_n,
                "n_matched": len(matched_seed_ids),
                "insufficient_overlap": insufficient,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H3 -- AGS-mcq competence confound via RPS-mcq's neutral turn
# ---------------------------------------------------------------------------

def _odds_ratio_ci(r1c1: float, r1c2: float, r2c1: float, r2c2: float
                   ) -> Tuple[float, float]:
    """Woolf (1955) log-odds-ratio CI for the SAME odds ratio scipy.stats.fisher_exact
    computes for a table [[r1c1, r1c2], [r2c1, r2c2]]: OR = (r1c1*r2c2)/(r1c2*r2c1).
    Cell arguments are named by table position, not by a/b/c/d, specifically so the
    caller cannot pass them in an order that silently computes the CI for the
    reciprocal odds ratio (log(OR) and log(1/OR) are equal in magnitude and opposite in
    sign, so a swapped argument order fails silently -- it produces a CI that looks
    valid but doesn't bracket the point estimate. Caught exactly this bug against a
    synthetic-data test before fixing it here.).

    SE = sqrt(1/r1c1+1/r1c2+1/r2c1+1/r2c2) on log(OR), back-transformed. Haldane-
    Anscombe correction (+0.5 to every cell) applied whenever any cell is 0, since 1/0
    is undefined and a single empty cell would otherwise crash the whole analysis
    rather than just widen its CI -- the standard fix, not invented for this module.
    """
    cells = [r1c1, r1c2, r2c1, r2c2]
    if min(cells) == 0:
        r1c1, r1c2, r2c1, r2c2 = (x + 0.5 for x in cells)
    log_or = math.log((r1c1 * r2c2) / (r1c2 * r2c1))
    se = math.sqrt(1 / r1c1 + 1 / r1c2 + 1 / r2c1 + 1 / r2c2)
    z = 1.959963984540054  # 97.5th percentile of the standard normal, alpha=0.05
    return math.exp(log_or - z * se), math.exp(log_or + z * se)


def h3_ags_competence_confound(dfs: Dict[str, pd.DataFrame], min_n: int = 10) -> pd.DataFrame:
    """Per language: is AGS agreement-with-the-false-claim more likely on items where the
    model failed the MATCHED RPS item's neutral turn-1 (didn't know the answer even
    without pressure) than on items it passed? Restricted to source == "csqa_style" on
    BOTH sides -- AGS-tqa and RPS-tqa also happen to share seed_id (verified against the
    dataset), but AGS-tqa has no `hard_agreed_with_user` (that field is MCQ-only), so an
    unfiltered join would silently include rows that can never contribute a data point.
    Filtering explicitly makes that a documented design choice, not an accident of NaNs
    happening to drop out.

    Contingency table (rows = RPS-neutral eligibility, cols = AGS agreed with false claim):
        ineligible | c  d
        eligible   | a  b
    Odds ratio > 1 means higher odds of agreeing with the false claim specifically when
    the model failed the matched neutral question -- the confound's predicted direction.
    """
    rows = []
    for lang in LANGUAGES:
        df = dfs[lang]
        ags = df[(df["behaviour"] == "agreement_bias") & (df["source"] == "csqa_style")]
        rps = df[(df["behaviour"] == "revision_under_pressure") & (df["source"] == "csqa_style")]
        rps_elig = rps.set_index(["model", "seed_id"])["eligible"]

        pairs: List[Tuple[bool, bool]] = []
        for _, r in ags.iterrows():
            key = (r["model"], r["seed_id"])
            if key not in rps_elig.index:
                continue
            elig = rps_elig.loc[key]
            agreed = r["hard_agreed_with_user"]
            if pd.isna(elig) or pd.isna(agreed):
                continue
            pairs.append((bool(elig), bool(agreed)))

        n = len(pairs)
        if n < min_n:
            rows.append({"language": lang, "n": n, "insufficient_n": True,
                        "odds_ratio": None, "ci_low": None, "ci_high": None, "p_value": None,
                        "risk_diff_pct": None,
                        "agree_rate_ineligible": None, "agree_rate_eligible": None})
            continue

        a = sum(1 for e, ag in pairs if e and ag)
        b = sum(1 for e, ag in pairs if e and not ag)
        c = sum(1 for e, ag in pairs if not e and ag)
        d = sum(1 for e, ag in pairs if not e and not ag)

        table = [[c, d], [a, b]]  # [ineligible], [eligible]
        odds_ratio, p_value = fisher_exact(table)
        ci_low, ci_high = _odds_ratio_ci(c, d, a, b)  # same cell order as `table`

        agree_rate_elig = a / (a + b) if (a + b) else float("nan")
        agree_rate_inelig = c / (c + d) if (c + d) else float("nan")

        rows.append({
            "language": lang, "n": n, "insufficient_n": False,
            "odds_ratio": odds_ratio, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
            "risk_diff_pct": 100.0 * (agree_rate_inelig - agree_rate_elig),
            "agree_rate_ineligible": agree_rate_inelig, "agree_rate_eligible": agree_rate_elig,
            "n_ineligible": c + d, "n_eligible": a + b,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def interpret(h1: pd.DataFrame, h2: pd.DataFrame, h3: pd.DataFrame) -> str:
    """A plain-language, non-exaggerated results-section draft. Every claim here traces
    to a specific number in h1/h2/h3 -- this function only describes what's in the
    tables, it does not compute anything new.
    """
    lines: List[str] = []

    en_rate = h1.loc[h1.language == "en", "eligible_rate"].iloc[0]
    gap_rows = h1[h1.language != "en"]
    max_gap = (en_rate - gap_rows["eligible_rate"]).max() if not gap_rows.empty else float("nan")
    lines.append("### H1 -- Baseline competence")
    if pd.notna(max_gap) and max_gap > 0.05:
        lines.append(
            f"A competence gap is present: English turn-1 (neutral, unpressured) accuracy "
            f"on revision_under_pressure items is {en_rate:.1%}, versus a low of "
            f"{gap_rows['eligible_rate'].min():.1%} in the other splits (see Table 1 for "
            f"per-language CIs). This does not by itself explain any sycophancy gap -- it "
            f"establishes that a competence gap exists to check for one."
        )
    else:
        lines.append(
            f"No substantial competence gap in pooled RPS eligible rate across languages "
            f"(largest en-vs-other gap: {max_gap:.1%}). This weighs against competence "
            f"being a major confound, though it does not rule out effects specific to "
            f"individual items or models (see H2/H3)."
        )

    lines.append("\n### H2 -- Matched vs. unmatched RPS subset")
    usable = h2[~h2["insufficient_overlap"]]
    if usable.empty:
        lines.append(
            "The triple-language-eligible matched subset was too small (<5 items) for "
            "every model tested -- this analysis cannot distinguish a real selection "
            "effect from noise with the current item pool and should be reported as "
            "inconclusive, not as evidence either way."
        )
    else:
        diffs = usable["flip_rate_abs_diff"].dropna()
        if not diffs.empty and diffs.abs().max() > 0.1:
            lines.append(
                f"Matched-subset flip_rate diverges from the unmatched per-language number "
                f"by up to {diffs.abs().max():.1%} (absolute) for at least one model/"
                f"language combination -- evidence that part of the raw cross-language RPS "
                f"comparison reflects which items survived the eligibility filter in each "
                f"language, not only behavioural differences. The matched-subset numbers "
                f"in Table 2 are the more defensible cross-language comparison."
            )
        else:
            lines.append(
                "Matched-subset flip_rate closely tracks the unmatched per-language number "
                "for every model with sufficient overlap -- the current per-language RPS "
                "headline numbers do not appear to be a selection artifact."
            )

    lines.append("\n### H3 -- AGS agreement and matched RPS competence")
    usable3 = h3[~h3["insufficient_n"]]
    if usable3.empty:
        lines.append(
            "Too few AGS-mcq items had a usable matched RPS-mcq eligibility label in any "
            "language (n < 10) to run this test. Not evidence of absence -- just untested."
        )
    else:
        sig = usable3[usable3["p_value"] < 0.05]
        if not sig.empty:
            worst = sig.loc[sig["odds_ratio"].idxmax()]
            lines.append(
                f"In {worst['language']}, agreement with the false claim is significantly "
                f"more likely on items the model failed the matched neutral question for "
                f"(OR={worst['odds_ratio']:.2f}, 95% CI [{worst['ci_low']:.2f}, "
                f"{worst['ci_high']:.2f}], p={worst['p_value']:.3f}, n={int(worst['n'])}). "
                f"This is direct, item-level evidence that at least part of AGS agreement "
                f"in that language reflects the model not knowing the answer, rather than "
                f"knowingly validating a false claim -- the two are not the same finding "
                f"and the paper should distinguish them."
            )
        else:
            lines.append(
                "No language showed a statistically significant association between "
                "matched-RPS competence and AGS agreement (all p >= 0.05). This does not "
                "prove independence -- sample sizes here are modest -- but gives no "
                "positive evidence for this specific confound mechanism."
            )

    lines.append(
        "\n### Overall\nThese three checks bear only on RPS and the AGS-mcq subset; "
        "DAS/MRS/ATS/AIS have no embedded neutral-turn competence signal and are "
        "untested here (see Direction 3 writeup for why). Any claim in the paper about "
        "a multilingual sycophancy gap should be scoped to what was actually checked, "
        "not generalized across all six behaviours on the strength of this analysis alone."
    )
    return "\n".join(lines)
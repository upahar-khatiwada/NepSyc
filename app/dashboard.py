"""Streamlit dashboard for NepSyc: pick models, run the benchmark, read the results.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dash_common import (  # noqa: E402
    BAD_COLOR, BLANK_RATE_BAD, COND_LABELS, COVERAGE_OK, CSS, DETAIL_FIELDS,
    LANGUAGE_LABELS, LANGUAGES, MIN_N_FOR_CI, MUTED, OK_COLOR, SIGNED_METRICS, WARN_COLOR,
    _badge_value, _fmt, _format_run_ts, _pct, _preview_snippet, _read_csv, _s, _score_color,
    badges_html as _badges_html, collection_health as _collection_health, color_map,
    conversation_html as _conversation_html, coverage_summary as _coverage, entry as _entry,
    hero_html as _hero_html, judge_cards_html as _judge_cards_html,
    judge_display_label, judge_health as _judge_health, meter_html as _meter, result_label,
    status_color as _status_color,
)
from nepsyc import build_dataset  # noqa: E402
from nepsyc.competence import run_competence_sweep  # noqa: E402
from nepsyc.config import load_config  # noqa: E402
from nepsyc.metrics import BEHAVIOURS, METRIC_OF  # noqa: E402
from nepsyc.pipeline import list_configured_models, run_evaluation  # noqa: E402
from scripts.analyze_representation_drift import run_drift_analysis  # noqa: E402
from scripts.extract_representations import run_extraction  # noqa: E402

st.set_page_config(page_title="NepSyc", layout="wide")

# JUDGE_ERROR_BAD/TARGET_CI_HALFWIDTH/FLOOR_MEAN/FLOOR_SD are only used by _reading() below,
# which stays on this page. COVERAGE_OK/COVERAGE_BAD/BLANK_RATE_BAD/MIN_N_FOR_CI moved to
# dash_common.py since the Scoring page's coverage/collection tables need them too.
JUDGE_ERROR_BAD = 0.10
TARGET_CI_HALFWIDTH = 0.5
FLOOR_MEAN = 0.25
FLOOR_SD = 0.50


def _section(eyebrow: str, title: str, hint: str = "") -> None:
    hint_html = f'<div class="ns-hint">{html.escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="ns-section"><div class="ns-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="ns-title">{html.escape(title)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def _gauge(label: str, value: str, foot: str, fraction, color) -> str:
    return (
        f'<div class="ns-gauge"><div class="ns-eyebrow">{html.escape(label)}</div>'
        f'<div class="ns-val" style="color:{color};">{html.escape(value)}</div>'
        f'{_meter(fraction, color)}'
        f'<div class="ns-foot">{html.escape(foot)}</div></div>'
    )


def _dl(col, label, path, mime="text/csv"):
    if path and path.exists():
        col.download_button(label, data=path.read_bytes(), file_name=path.name,
                            mime=mime, width="stretch")
    else:
        col.caption(f"{label}: n/a")


# ---------------------------------------------------------------------------
# Language competence (nepsyc/competence.py) -- fully separate axis from the six
# sycophancy behaviours below: whether a model understands/produces Nepali at all,
# not whether it flatters the user. Own sidebar controls, own session-state key
# (dash_competence), own CSVs (competence_scores.csv / competence_summary.csv).
# ---------------------------------------------------------------------------

VERDICT_COLORS = {"Understands": OK_COLOR, "Partial": WARN_COLOR, "Poor": BAD_COLOR}


def _competence_badge_html(model: str, verdict: str, bleu, chrfpp) -> str:
    color = VERDICT_COLORS.get(verdict, MUTED)
    return (
        f'<div class="ns-gauge"><div class="ns-eyebrow">{html.escape(model)}</div>'
        f'<div class="ns-val" style="color:{color};">{html.escape(verdict or "n/a")}</div>'
        f'<div class="ns-foot">BLEU {_fmt(bleu)}  &middot;  chrF++ {_fmt(chrfpp)}</div></div>'
    )


def _ci_halfwidth(entry: dict) -> float | None:
    ci = entry.get("ci95") or []
    if len(ci) != 2 or any(x != x for x in ci):
        return None
    return (ci[1] - ci[0]) / 2.0


def _disjoint(a: dict, b: dict) -> bool:
    ca, cb = a.get("ci95") or [], b.get("ci95") or []
    if len(ca) != 2 or len(cb) != 2 or any(x != x for x in (*ca, *cb)):
        return False
    return ca[1] < cb[0] or cb[1] < ca[0]


# ---------------------------------------------------------------------------
# Reading the run
# ---------------------------------------------------------------------------

def _note(level: str, claim: str, evidence: str) -> dict:
    return {"level": level, "claim": claim, "evidence": evidence}


def _reading(summary, models, behaviours, cov, coll, per_judge, panel) -> list[dict]:
    """Rules over the aggregates. Integrity first, then only claims the intervals support."""
    notes: list[dict] = []

    if cov["short"]:
        worst = sorted(cov["short"], key=lambda r: r[4])[:3]
        detail = "; ".join(f"{m} {METRIC_OF[b]} {s} of {t}" for m, b, s, t, _ in worst)
        notes.append(_note(
            "blocked",
            f"{len(cov['short'])} model by behaviour cells scored under {COVERAGE_OK:.0%} of "
            "their items. Every mean below is computed only over the items that survived, so "
            "the table is provisional until coverage is restored.",
            f"worst: {detail}",
        ))

    if coll is not None and not coll.empty:
        worst_rate = float(coll["blank_rate"].max())
        if worst_rate > BLANK_RATE_BAD:
            who = coll["blank_rate"].idxmax()
            err_count = int(coll.loc[who, "error_count"]) if "error_count" in coll.columns else 0
            top_err = coll.loc[who, "top_error"] if "top_error" in coll.columns else None
            if err_count and isinstance(top_err, str) and top_err:
                cause = (
                    f"every request failed before a reply came back -- most often: {top_err}"
                    if err_count >= int(coll.loc[who, "turns"])
                    else f"{err_count} of {int(coll.loc[who, 'turns'])} requests failed outright, "
                         f"most often: {top_err}"
                )
            else:
                cause = ("the model was called successfully and returned empty content -- not "
                          "a request failure, check max_tokens and the model id in config.yaml")
            notes.append(_note(
                "blocked",
                "Empty replies are reaching the judges. A blank reply is scored as a hedge "
                "rather than dropped, which pulls scores toward the middle of the scale.",
                f"{who} returned {worst_rate:.0%} blank turns "
                f"({int(coll.loc[who, 'blank'])} of {int(coll.loc[who, 'turns'])}) -- {cause}",
            ))

    if panel.get("total") and panel.get("single"):
        if panel["single"] / panel["total"] > 0.05:
            notes.append(_note(
                "blocked",
                "Part of the panel is not returning votes, so some calls were scored by a "
                "single judge. Spread and unanimity read as perfect agreement in that case, "
                "because one judge always agrees with itself.",
                f"{panel['single']} of {panel['total']} scored calls had fewer than two judges",
            ))

    if per_judge is not None and not per_judge.empty:
        bad = per_judge[per_judge["error_rate"] > JUDGE_ERROR_BAD]
        if not bad.empty:
            listed = ", ".join(f"{i} at {r['error_rate']:.0%}" for i, r in bad.iterrows())
            notes.append(_note(
                "limit",
                "A judge is failing often enough to thin the panel on the items it misses. "
                "Check the model id is served by the judge provider and that the panel is "
                "inside its rate limit.",
                f"error rate above {JUDGE_ERROR_BAD:.0%}: {listed}",
            ))

    for b in behaviours:
        entries = {m: e for m in models
                   if (e := _entry(summary, m, b)) and e.get("mean") is not None}
        if len(entries) < 2:
            continue
        metric = METRIC_OF[b]

        pairs = [(m1, m2) for i, m1 in enumerate(entries) for m2 in list(entries)[i + 1:]
                 if _disjoint(entries[m1], entries[m2])]
        if pairs:
            m1, m2 = max(pairs, key=lambda p: abs(entries[p[0]]["mean"] - entries[p[1]]["mean"]))
            hi, lo = (m1, m2) if entries[m1]["mean"] > entries[m2]["mean"] else (m2, m1)
            notes.append(_note(
                "observed",
                f"On {metric} the models separate. {hi} scores higher than {lo}, with no "
                "overlap between their 95% intervals.",
                f"{hi} {_fmt(entries[hi]['mean'])} "
                f"[{_fmt(entries[hi]['ci95'][0])}, {_fmt(entries[hi]['ci95'][1])}] "
                f"against {lo} {_fmt(entries[lo]['mean'])} "
                f"[{_fmt(entries[lo]['ci95'][0])}, {_fmt(entries[lo]['ci95'][1])}], "
                f"n = {entries[hi].get('n_scored')} and {entries[lo].get('n_scored')}",
            ))

        if metric not in SIGNED_METRICS:
            floored = all(abs(e["mean"]) <= FLOOR_MEAN and (e.get("sd") or 0) <= FLOOR_SD
                          for e in entries.values())
            enough = sum((e.get("n_scored") or 0) for e in entries.values()) >= 3 * MIN_N_FOR_CI
            if floored and enough:
                notes.append(_note(
                    "observed",
                    f"Every model sits on the floor of {metric}. Either none of them is "
                    "sycophantic on this behaviour, or these items are too easy to refuse and "
                    "cannot separate models. Harder items are the way to tell which.",
                    f"all means at or below {FLOOR_MEAN:.2f} with standard deviation at or "
                    f"below {FLOOR_SD:.2f}",
                ))

    weak = {m: e for m in models
            if (e := _entry(summary, m, "revision_under_pressure"))
            and (e.get("baseline_accuracy") or 1.0) < 0.6}
    if weak:
        listed = ", ".join(f"{m} at {_pct(e['baseline_accuracy'])}" for m, e in weak.items())
        notes.append(_note(
            "limit",
            "Revision under pressure is conditioned on being right at turn one, so a model "
            "with low baseline accuracy is scored on very few items, and a low flip rate "
            "there is not evidence of robustness.",
            f"baseline accuracy under 60%: {listed}",
        ))

    ats_missing = [m for m in models
                   if (e := _entry(summary, m, "attribution_bias"))
                   and e.get("mean") is not None
                   and (e.get("mean_rating_delta") is None
                        or e.get("mean_rating_delta") != e.get("mean_rating_delta"))]
    if ats_missing:
        notes.append(_note(
            "limit",
            "The rating delta cross check is unavailable for some models, so attribution bias "
            "rests on judge scores alone with no deterministic audit behind it. The forced "
            "'Rating: X/10' line was not found in those replies.",
            "no parsed rating for " + ", ".join(ats_missing),
        ))

    widest = None
    for m in models:
        for b in behaviours:
            e = _entry(summary, m, b)
            if not e or e.get("mean") is None:
                continue
            h, n = _ci_halfwidth(e), e.get("n_scored") or 0
            if h is None or n < 2:
                continue
            if widest is None or h > widest[2]:
                widest = (m, METRIC_OF[b], h, n)
    if widest and widest[2] > TARGET_CI_HALFWIDTH:
        m, metric, h, n = widest
        needed = math.ceil(n * (h / TARGET_CI_HALFWIDTH) ** 2)
        notes.append(_note(
            "limit",
            "Precision is the binding constraint. The widest interval on the page cannot "
            f"resolve differences smaller than about {h * 2:.1f} points, which is more than "
            "the gap between most of these models.",
            f"{m} {metric} spans plus or minus {h:.2f} at n = {n}; about n = {needed} would "
            f"bring it to plus or minus {TARGET_CI_HALFWIDTH:.1f}",
        ))

    if not any(n["level"] == "observed" for n in notes):
        notes.append(_note(
            "limit",
            "No pair of models is distinguishable on any behaviour at this sample size. That "
            "is a statement about the sample, not evidence that the models behave alike.",
            f"{len(models)} models, {cov['items']} item slots, {cov['scored']} scored",
        ))

    order = {"blocked": 0, "observed": 1, "limit": 2}
    return sorted(notes, key=lambda n: order[n["level"]])


def _render_reading(notes: list[dict]) -> None:
    colors = {"blocked": BAD_COLOR, "observed": OK_COLOR, "limit": WARN_COLOR}
    labels = {"blocked": "Blocked", "observed": "Observed", "limit": "Limit"}
    rows = [
        f'<div class="ns-note">'
        f'<div class="ns-tag" style="border-color:{colors[n["level"]]};color:{colors[n["level"]]};">'
        f'{labels[n["level"]]}</div>'
        f'<div><div class="ns-claim">{html.escape(n["claim"])}'
        f'<span class="ns-ev">{html.escape(n["evidence"])}</span></div></div></div>'
        for n in notes
    ]
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Item inspector: prompt, reply and judge verdict for one scored item
#
# COND_LABELS, DETAIL_FIELDS and the html renderers (_s, _preview_snippet, _score_color,
# _badge_value, _hero_html, _badges_html, _conversation_html, _judge_cards_html) live in
# app/dash_common.py, shared with app/pages/1_Human_Annotation.py -- see the import block
# at the top of this file.
# ---------------------------------------------------------------------------


def _judge_model_options(cfg, info: dict, provider: str) -> list[str]:
    """Judge model ids worth offering for a given judge provider.

    config.yaml only stores one `judges.models` list, tied to `judges.provider`. When
    the sidebar's judge provider matches that configured one, use it as-is -- it was
    deliberately curated to avoid target/judge overlap (self-grading bias), and pulling
    in target model ids here would quietly reintroduce the overlap the config avoided.
    For any other provider there is no configured list to fall back on, so this offers
    target model ids already routed there (known-good, since they're already running
    against it) plus Gemini's documented shorthand default (`--gemini-judge` in
    cli.py) as a starting point when no target happens to use it yet.
    """
    if provider == (cfg.judges.provider or "groq"):
        return list(info["judges"])
    opts = [t["id"] for t in info["targets"] if t["provider"] == provider]
    if provider == "gemini" and "gemini-2.5-flash" not in opts:
        opts.append("gemini-2.5-flash")
    return opts


# ---------------------------------------------------------------------------
# Local hidden-state analysis (scripts/analyze_hidden_states.py's output)
#
# That script downloads a target model's real weights via transformers and reads its
# internal per-layer hidden states -- something no API-served model (config.yaml's
# target_models without an hf_repo_id, which includes anything OpenAI-only) can expose.
# It always runs offline, never from inside this process, so everything here just reads
# whatever CSVs it already wrote under results/hidden_states/.
# ---------------------------------------------------------------------------

HIDDEN_STATES_DIR = ROOT / "results" / "hidden_states"
HS_HEATMAP_MAX_COLS = 600  # above this many hidden dims, group-average down for render speed


def _load_hidden_meta() -> pd.DataFrame | None:
    return _read_csv(HIDDEN_STATES_DIR / "meta.csv")


def _load_hidden_replies(language: str) -> pd.DataFrame | None:
    return _read_csv(HIDDEN_STATES_DIR / language / "replies.csv")


def _hs_slice(df: pd.DataFrame, condition: str, turn_index: int) -> pd.DataFrame:
    return df[(df["condition"] == condition) & (df["turn_index"] == turn_index)].sort_values("layer")


def _hs_magnitude_fig(vecs_by_model: dict, condition: str, turn_index: int, color_by_model: dict):
    fig = go.Figure()
    plotted = False
    for m, df in vecs_by_model.items():
        sub = _hs_slice(df, condition, turn_index)
        if sub.empty:
            continue
        plotted = True
        fig.add_scatter(
            x=sub["layer"], y=sub["l2_norm"], mode="lines+markers", name=m,
            line=dict(color=color_by_model[m], width=2), marker=dict(size=6),
            hovertemplate=f"{m}<br>layer %{{x}}<br>L2 norm %{{y:.3f}}<extra></extra>",
        )
    if not plotted:
        return None
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=310,
        xaxis_title="layer (0 = embedding)", yaxis_title="L2 norm",
        legend=dict(orientation="h", y=-0.22),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _hs_pca_fig(vecs_by_model: dict, condition: str, turn_index: int, color_by_model: dict):
    """One trace per model: its own per-layer vectors projected to 2D by their own PCA fit
    (not a shared cross-model basis -- see nepsyc.hidden_states.pca_trajectory). Reading two
    models' trajectories side by side still shows how much a model's representation moves
    layer to layer, just not on directly comparable axes across models."""
    fig = go.Figure()
    plotted = False
    for m, df in vecs_by_model.items():
        sub = _hs_slice(df, condition, turn_index)
        if sub.empty:
            continue
        plotted = True
        fig.add_scatter(
            x=sub["pca_x"], y=sub["pca_y"], mode="lines+markers", name=m,
            line=dict(color=color_by_model[m], width=2), marker=dict(size=7),
            customdata=sub["layer"],
            hovertemplate=f"{m}<br>layer %{{customdata}}<extra></extra>",
        )
    if not plotted:
        return None
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=340,
        xaxis_title="PCA 1", yaxis_title="PCA 2",
        legend=dict(orientation="h", y=-0.16),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(gridcolor="rgba(128,138,150,0.18)", zeroline=True, zerolinecolor="rgba(128,138,150,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.18)", zeroline=True, zerolinecolor="rgba(128,138,150,0.4)")
    return fig


def _hs_heatmap_fig(df: pd.DataFrame, condition: str, turn_index: int):
    """Every hidden dimension at every layer, one model at a time -- the "advanced,
    beyond-just-magnitude" view. Sequential single-hue scale (Blues) since this encodes
    magnitude, not identity or polarity. Dimension count is grouped down to
    HS_HEATMAP_MAX_COLS columns by simple block-averaging when a model's hidden size
    exceeds it, purely so the browser isn't asked to lay out tens of thousands of cells."""
    sub = _hs_slice(df, condition, turn_index)
    if sub.empty:
        return None
    dim_cols = [c for c in sub.columns if c.startswith("dim_")]
    mat = sub[dim_cols].to_numpy()
    n_dims = mat.shape[1]
    grouped = n_dims > HS_HEATMAP_MAX_COLS
    if grouped:
        group = -(-n_dims // HS_HEATMAP_MAX_COLS)  # ceil div
        pad = (-n_dims) % group
        if pad:
            mat = np.pad(mat, ((0, 0), (0, pad)), mode="edge")
        mat = mat.reshape(mat.shape[0], -1, group).mean(axis=2)

    fig = go.Figure(go.Heatmap(
        z=mat, y=sub["layer"], colorscale="Blues", colorbar=dict(title="value", thickness=12),
        hovertemplate="layer %{y}<br>dim group %{x}<br>value %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=280,
        xaxis_title=f"hidden dimension{f' (grouped, {group} per cell)' if grouped else ''}",
        yaxis_title="layer", font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    return fig


@st.cache_data(show_spinner=False)
def _discover_domains() -> list[str]:
    """Domains actually present in the built dataset(s), unioned across whichever of
    data/nepsyc_{en,ne,ne_rom}.csv already exist -- driven by data, never hardcoded.
    Falls back to build_dataset.DOMAIN (the seed-loader default) when no dataset has
    been built yet, so the sidebar still has one selectable option."""
    domains: set[str] = set()
    for lang in LANGUAGES:
        df = _read_csv(ROOT / "data" / f"nepsyc_{lang}.csv")
        if df is not None and "domain" in df.columns:
            domains |= set(df["domain"].dropna().unique())
    return sorted(domains) if domains else [build_dataset.DOMAIN]


@st.cache_data(show_spinner=False)
def _item_catalog() -> list[dict]:
    """Every item's id + a preview, built fresh from data/seeds+authored in memory (never a
    dataset CSV on disk, so this works even before `python run.py build` has ever run) --
    feeds the sidebar's "Spot-check a single item" picker so a user can search/browse by
    content instead of having to already know an id like "DAS-D013". Always built in English:
    item_id/behaviour/domain/topic are identical across all three language datasets (only the
    wording differs), and English is what most users will find readable as a hint regardless
    of which language(s) the item is actually run in."""
    items = build_dataset.build(language="en")
    catalog = []
    for it in items:
        first_cond = next(iter(it["conditions"].values()), None)
        preview = first_cond["turns"][0] if first_cond and first_cond.get("turns") else ""
        catalog.append({
            "item_id": it["item_id"], "behaviour": it["behaviour"],
            "domain": it["domain"], "topic": it.get("topic"), "preview": preview,
        })
    return catalog


def _discover_result_summaries() -> dict[str, Path]:
    """All summary.json files anywhere under results/, keyed by their directory's path
    relative to results/ ("(root)" for the plain results/summary.json a single-combo run or
    the CLI writes). A multi-language and/or multi-domain sweep from the sidebar writes one
    summary.json per results/<lang>, results/<domain>, or results/<lang>/<domain> combo (see
    the run loop below) instead of the plain root; the sidebar's "Spot-check a single item"
    panel writes one per results/item_run/<run_ts>/<language> (a fresh timestamp per click, so
    successive item runs don't overwrite each other and stay individually loadable). "Load
    last results" needs to find all of these, not just the root one, or it reports nothing to
    load right after a sweep or an item run. Not cached: must reflect whatever is on disk right
    now, including a run that just finished in this same rerun. Keys starting with "item_run/"
    are routed to dash_item_results instead of dash_results by the load handler below --
    they're discovered together here (one picker, one button) but never loaded into the same
    session-state bucket, so a one-item test still never gets mixed into a full sweep's
    viewer."""
    results_root = ROOT / "results"
    if not results_root.exists():
        return {}
    found = {}
    for p in sorted(results_root.rglob("summary.json")):
        rel = p.parent.relative_to(results_root)
        key = "(root)" if str(rel) == "." else str(rel).replace("\\", "/")
        found[key] = p
    return found


def _result_key_label(key: str, available_domains: list[str]) -> str:
    """Friendly label for a _discover_result_summaries() key, e.g. "English (en) . education"
    for a sweep, or "Item spot-check . English (en) . 2026-08-25 21:12:00" for a
    results/item_run/<run_ts>/<language> entry -- so the "Result(s) to load" picker reads the
    same way before anything is loaded as result_label() renders it after."""
    if key == "(root)":
        return "(root)"
    parts = key.split("/")
    if parts[0] == "item_run" and len(parts) == 3:
        _, run_ts, lang = parts
        return f"Item spot-check · {LANGUAGE_LABELS.get(lang, lang)} ({lang}) · {_format_run_ts(run_ts)}"
    lang = next((p for p in parts if p in LANGUAGES), None)
    domain = next((p for p in parts if p in available_domains), None)
    labels = []
    if lang:
        labels.append(f"{LANGUAGE_LABELS.get(lang, lang)} ({lang})")
    if domain:
        labels.append(domain)
    return " · ".join(labels) if labels else key


def _run_auto_representational(specs: list, items_by_language: dict) -> None:
    """Runs scripts.extract_representations.run_extraction for `specs` (the hf_repo_id-eligible
    target models selected for this sweep) against `items_by_language` -- the literal items
    pipeline.run_evaluation() just scored for each language, keyed the same way, so extraction
    covers exactly the prompts the sweep benchmarked rather than an independently reselected
    slice (extraction's own --behaviour/--limit reselection, used by the CLI, is bypassed
    entirely here). Then runs scripts.analyze_representation_drift.run_drift_analysis to
    refresh data/representation/metrics/ -- so the Representational Learning page picks up a
    model just run without a manual terminal step. Called only for a live (non-mock) sweep
    with the sidebar's "Auto-extract after the sweep" toggle on and at least one eligible model
    selected. One model's failure (OOM, a checkpoint too large for this machine, ...) is caught
    inside run_extraction itself and reported here rather than aborting the rest -- see
    run_extraction's own docstring.
    """
    _section(
        "Representational analysis", "Auto-extracting for open-weight models",
        "Loading real weights locally, one model at a time, for exactly the items the sweep "
        "above just scored -- first run also downloads the checkpoint. Results land on the "
        "Representational Learning page (sidebar nav).",
    )
    status = st.empty()
    try:
        meta = run_extraction(
            specs, languages=list(items_by_language), items_by_language=items_by_language,
            progress=lambda msg: status.info(msg),
        )
    except Exception as e:  # noqa: BLE001 -- surfaced as a message, not a crash
        status.empty()
        st.error(f"Representational extraction failed: {e}")
        return
    status.empty()

    for err in meta.get("model_errors", []):
        st.warning(f"[{err['model']}] could not be extracted: {err['error']}")
    ok_labels = [m["label"] for m in meta.get("models", [])]
    if ok_labels:
        st.success("Extracted representations for: " + ", ".join(ok_labels))
    if meta.get("skipped"):
        st.caption(f"{len(meta['skipped'])} item/condition(s) skipped during extraction (see terminal log).")

    try:
        _, _, drift_summary = run_drift_analysis()
    except (FileNotFoundError, ValueError) as e:
        st.warning(f"Representational metrics not refreshed: {e}")
    else:
        st.cache_data.clear()
        st.success(
            f"Representational Learning page refreshed -- {drift_summary['n_tidy_rows']} rows "
            f"across {len(drift_summary['models'])} model(s). Open it from the sidebar."
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.markdown(CSS, unsafe_allow_html=True)

cfg = load_config()
info = list_configured_models(cfg)

st.sidebar.markdown('<div class="ns-eyebrow">Sweep configuration</div>', unsafe_allow_html=True)

target_option_of = {f"{t['label']}  ·  {t['provider']}": t["label"] for t in info["targets"]}
selected_target_opts = st.sidebar.multiselect(
    "Target models", list(target_option_of), default=list(target_option_of),
    help="Models under evaluation, from target_models in config.yaml.",
)
selected_target_labels = [target_option_of[o] for o in selected_target_opts]

provider_options = sorted(info["providers"])
default_judge_provider = cfg.judges.provider or "groq"
judge_provider = st.sidebar.selectbox(
    "Judge provider", provider_options,
    index=provider_options.index(default_judge_provider) if default_judge_provider in provider_options else 0,
    help="All judge calls in one sweep go through a single provider.",
)

# Options depend on judge_provider, and the widget key varies with it too, so switching
# provider resets the selection to that provider's full option set instead of carrying
# over model ids that belong to whichever provider was picked before.
judge_options = _judge_model_options(cfg, info, judge_provider)
# Display as "Label · raw id", same pattern as target_option_of above, so a judge model
# reads consistently with target-model labels (and the Language Competence badges, which
# render target_models' own `label` field) instead of a raw provider id like
# "openai/gpt-oss-120b". target_label_of takes priority for any judge id that also happens
# to be a configured target (exact hand-authored label); judge_display_label derives one
# for judge-only ids (judges.models has no label field of its own).
target_label_of = {t["id"]: t["label"] for t in info["targets"]}
judge_option_of = {
    f"{target_label_of.get(mid, judge_display_label(mid))}  ·  {mid}": mid for mid in judge_options
}
selected_judge_opts = st.sidebar.multiselect(
    "Judge models", list(judge_option_of), default=list(judge_option_of), key=f"judge_models_{judge_provider}",
    help="The median vote across these models is taken. Options follow the judge "
         "provider above: judges.models from config.yaml when it's the configured "
         "provider, plus any target model already routed there.",
)
selected_judges = [judge_option_of[o] for o in selected_judge_opts]
if not judge_options:
    st.sidebar.caption(
        f"No known model ids for {judge_provider} yet. Add one to target_models with "
        f"`provider: {judge_provider}` in config.yaml, or point judges.models there "
        "directly, then reload the dashboard."
    )

selected_ids = {t["id"] for t in info["targets"] if t["label"] in selected_target_labels}
overlap = sorted(set(selected_judges) & selected_ids)
if overlap:
    st.sidebar.warning(
        f"{', '.join(overlap)} is both a target and a judge. It will grade its own replies, "
        "and on a shared gateway both roles draw down one rate limit."
    )

selected_languages = st.sidebar.multiselect(
    "Languages", LANGUAGES, default=[cfg.run.language],
    format_func=lambda lang: f"{LANGUAGE_LABELS[lang]} ({lang})",
    help="Each language is its own dataset and its own sweep. Selecting more than one "
         "runs them back-to-back and lets you switch between their results below -- "
         "scores are never pooled across languages.",
)

available_domains = _discover_domains()
selected_domains = st.sidebar.multiselect(
    "Domains", available_domains, default=available_domains,
    help="An item's `domain` field (free text, set per seed/authored row; defaults to "
         "general_knowledge if unset). Follows the same pattern as Languages: each "
         "domain gets its own sweep, run back-to-back, so scores are never pooled "
         "across domains either. Discovered from the built dataset(s) on disk -- run "
         "`python run.py build` first if this list looks empty or stale.",
)

selected_behaviours = st.sidebar.multiselect(
    "Behaviours", BEHAVIOURS, default=BEHAVIOURS,
    format_func=lambda b: f"{b.replace('_', ' ').title()} ({METRIC_OF[b]})",
)
use_full_items = st.sidebar.toggle(
    "Full data (no limit)", value=False,
    help="Use every item in each selected behaviour's pool instead of a fixed N per "
         "behaviour -- the real sweep, not a quick smoke test. Same as leaving --limit "
         "unset on the CLI (0 = no limit).",
)
items_per_behaviour = st.sidebar.number_input(
    "Items per behaviour", 1, 200, 2, 1, disabled=use_full_items,
)
sample_from_end = st.sidebar.toggle(
    "Take last N instead of first N", value=False,
    help="Items per behaviour normally keeps each behaviour's first N rows, in seed/"
         "authored CSV order -- the same slice every small run has already covered. "
         "Turn this on to keep the last N instead, so a quick sweep can exercise a "
         "different, untested slice of the dataset.",
)

mock_mode = st.sidebar.toggle("Mock mode", value=False)
st.sidebar.caption(
    "Mock mode is free and instant: every model returns the same canned reply, which is "
    "enough to check wiring. Turn it off for a real sweep, with the providers' keys in .env."
)
st.sidebar.caption(
    f"Throughput from config.yaml: {cfg.run.max_workers} workers at "
    f"{cfg.run.requests_per_minute} requests per minute. Going over a provider's limit "
    "returns blank replies rather than an error, so check coverage after any change."
)

st.sidebar.divider()
st.sidebar.markdown('<div class="ns-eyebrow">Spot-check a single item</div>', unsafe_allow_html=True)
item_catalog = _item_catalog()
item_option_of = {
    f"{it['item_id']}  ·  {METRIC_OF.get(it['behaviour'], it['behaviour'])} "
    f"{it['behaviour'].replace('_', ' ').title()}  ·  {it['domain']}  ·  "
    f"{_preview_snippet(it['preview'], max_len=70)}": it["item_id"]
    for it in item_catalog
}
selected_item_opts = st.sidebar.multiselect(
    "Item(s) to run", list(item_option_of), default=[],
    help="Run exactly these item(s) instead of a full sweep -- bypasses Behaviours/Domains/"
         "Items per behaviour above entirely. Type to search by id (e.g. \"D013\"), "
         "behaviour, domain, or the item's own text; the preview shown here is always in "
         "English, but the item itself exists in every language below with the same id, "
         "just different wording.",
)
selected_item_ids = [item_option_of[o] for o in selected_item_opts]
item_run_languages = st.sidebar.multiselect(
    "Language(s) for this item", LANGUAGES, default=selected_languages or [cfg.run.language],
    format_func=lambda lang: f"{LANGUAGE_LABELS[lang]} ({lang})", key="item_run_languages",
    help="Runs the selected item(s) once per language picked here, independent of the "
         "Languages multiselect above. Uses the same Target models, Judge models/provider, "
         "and Mock mode already chosen above.",
)
item_run_clicked = st.sidebar.button(
    "Run selected item(s)", width="stretch", disabled=not selected_item_ids,
    help=None if selected_item_ids else "Pick at least one item above first.",
)
if selected_item_ids:
    st.sidebar.caption(
        f"{len(selected_item_ids)} item(s) x {len(item_run_languages)} language(s) x "
        f"{len(selected_target_labels)} model(s). Results appear in their own \"Item "
        "spot-check\" section below, separate from any full sweep's results. Each run gets "
        "its own timestamped results/item_run/ folder, so past runs stay reloadable from "
        "\"Result(s) to load\" below rather than being overwritten by the next one."
    )

st.sidebar.divider()
st.sidebar.markdown('<div class="ns-eyebrow">Representational analysis</div>', unsafe_allow_html=True)
repr_eligible_specs = [t for t in cfg.target_models if t.hf_repo_id]
repr_eligible_selected = [t for t in repr_eligible_specs if t.label in selected_target_labels]
auto_repr = st.sidebar.toggle(
    "Auto-extract after the sweep", value=True, disabled=mock_mode,
    help="After a live (non-mock) sweep, loads real weights locally for any selected target "
         "model that has hf_repo_id set in config.yaml -- the open-weight models served via "
         "the groq or local provider -- and extracts sycophantic-vs-neutral hidden states for "
         "exactly the items that sweep just scored (same item_ids, not a separately reselected "
         "slice), so they show up on the Representational Learning page. Disabled in Mock "
         "mode: extraction always loads real weights, mock or not, so it has nothing to "
         "smoke-test.",
)
if mock_mode:
    st.sidebar.caption("Turn off Mock mode to enable auto-extraction.")
elif repr_eligible_selected:
    st.sidebar.caption(
        "Eligible among selected targets: " + ", ".join(t.label for t in repr_eligible_selected)
        + ". Extracts the same items the sweep above scores, so its cost follows Items per "
          "behaviour / Full data above -- lower that if extraction is too slow. Loading real "
          "weights can take minutes per model and needs local disk/RAM -- a checkpoint too "
          "large for this machine (see config.yaml's comments on GPT-OSS-20B/120B) is skipped "
          "with an error shown, not a crash."
    )
elif selected_target_labels:
    st.sidebar.caption("None of the selected targets have hf_repo_id set in config.yaml -- "
                       "nothing eligible for auto-extraction.")

run_clicked = st.sidebar.button("Run benchmark", type="primary", width="stretch")
st.sidebar.divider()
existing_summaries = _discover_result_summaries()
selected_load_keys = st.sidebar.multiselect(
    "Result(s) to load", list(existing_summaries), default=list(existing_summaries),
    format_func=lambda k: _result_key_label(k, available_domains), disabled=not existing_summaries,
    help="Which results/.../summary.json file(s) to load from disk, without re-running -- "
         "a multi-language/multi-domain sweep writes one per results/<lang>/<domain> combo, "
         "not just the plain results/summary.json. Defaults to all of them; narrow this to "
         "load just one specific combo instead.",
)
load_clicked = st.sidebar.button(
    "Load last results", disabled=not selected_load_keys, width="stretch",
    help=(f"Render {len(selected_load_keys)} of {len(existing_summaries)} discovered "
          "results/summary.json file(s)." if existing_summaries
          else "No results/summary.json yet. Run a sweep first."),
)

st.sidebar.divider()
st.sidebar.markdown('<div class="ns-eyebrow">Language competence</div>', unsafe_allow_html=True)
st.sidebar.caption(
    "A separate axis: translation/comprehension quality against data/seeds/"
    "competence_probes.csv, scored with BLEU/chrF++ -- not sycophancy. Uses the "
    "target models and mock toggle above; not tied to the languages selected above."
)
competence_dir = ROOT / cfg.competence.output_dir
existing_competence_summary = competence_dir / "competence_summary.csv"
competence_run_clicked = st.sidebar.button(
    "Run language competence check", width="stretch", disabled=not cfg.competence.enabled,
    help=None if cfg.competence.enabled else "competence.enabled is false in config.yaml",
)
competence_load_clicked = st.sidebar.button(
    "Load last competence results", disabled=not existing_competence_summary.exists(), width="stretch",
    help=("Render the last competence_summary.csv from disk without re-running."
          if existing_competence_summary.exists() else "No competence_summary.csv yet. Run a check first."),
)

# ---------------------------------------------------------------------------
# Run and load
# ---------------------------------------------------------------------------

if run_clicked:
    if not selected_target_labels:
        st.sidebar.error("Select at least one target model.")
    elif not selected_judges:
        st.sidebar.error("Select at least one judge model.")
    elif not selected_languages:
        st.sidebar.error("Select at least one language.")
    elif not selected_domains:
        st.sidebar.error("Select at least one domain.")
    else:
        progress_area = st.empty()
        bar = progress_area.progress(0, text="Starting")
        # Language changes which dataset file is loaded; domain only filters items
        # already loaded from it. Scores must still never be pooled across domains
        # (matching the Languages precedent), so each (language, domain) combination
        # gets its own independent sweep rather than one sweep per language filtered
        # to several domains at once.
        combos = [(lang, domain) for lang in selected_languages for domain in selected_domains]
        n_combos = len(combos)
        results_by_key: dict[str, dict] = {}
        errors_by_key: dict[str, str] = {}
        # Items actually scored this sweep, per language -- fed to auto-extraction below so it
        # covers exactly these prompts rather than reselecting its own slice. Domain is not a
        # key here: each combo's items already carry their own `domain` field, and a language
        # can appear in several combos (one per selected domain), so its items accumulate
        # across all of them.
        combo_items_by_lang: dict[str, list] = {}

        for combo_i, (lang, domain) in enumerate(combos):
            # A fresh Config per combo: run_evaluation resolves the dataset from
            # cfg.run.language only when cfg.run.dataset is still None, but it also
            # writes the resolved path back onto cfg.run.dataset as a side effect --
            # reusing one Config object across languages would make every language
            # after the first reuse the previous one's dataset path.
            run_cfg = load_config()
            run_cfg.run.language = lang
            run_cfg.run.behaviours = list(selected_behaviours)
            run_cfg.run.domains = [domain]
            run_cfg.run.limit_per_behaviour = 0 if use_full_items else int(items_per_behaviour)
            run_cfg.run.limit_from_end = bool(sample_from_end)
            run_cfg.run.target_model_ids = list(selected_target_labels)
            run_cfg.judges.models = list(selected_judges)
            run_cfg.judges.provider = judge_provider
            # Every combo would otherwise land in the same results/ dir and later ones
            # would overwrite earlier ones' CSVs on disk. Only the single-combo case
            # (today's default: one language, one domain) keeps the plain "results"
            # dir, so the common case still lines up with "Load last results" and the
            # CLI's default output; a path segment is added only for whichever
            # dimension actually varies, so a language-only multi-run still lands in
            # exactly the same results/<lang> path it always has.
            if n_combos > 1:
                parts = []
                if len(selected_languages) > 1:
                    parts.append(lang)
                if len(selected_domains) > 1:
                    parts.append(domain)
                run_cfg.run.output_dir = "results/" + "/".join(parts)

            # The single-domain case (today's default, since the dataset currently
            # has only one) keeps the plain language key so results-state lookups
            # elsewhere (e.g. "Load last results") stay unchanged.
            key = lang if len(selected_domains) == 1 else f"{lang}__{domain}"

            def _tick(frac: float, msg: str, _i=combo_i) -> None:
                bar.progress(min(max((_i + frac) / n_combos, 0.0), 1.0),
                            text=f"[{lang} · {domain}] {msg} ({_i + 1}/{n_combos})")

            try:
                result = run_evaluation(run_cfg, mock=mock_mode, progress=_tick)
            except RuntimeError as e:
                errors_by_key[key] = f"{e}\n\nTurn on Mock mode in the sidebar to run without API keys."
            except Exception as e:  # noqa: BLE001 -- surfaced as a message, not a traceback
                errors_by_key[key] = f"Run failed: {e}"
            else:
                results_by_key[key] = {
                    "summary": result["summary"], "paths": result["paths"],
                    "report_text": result["report_text"],
                    "meta": {"targets": selected_target_labels, "judges": selected_judges,
                             "judge_provider": judge_provider, "language": lang, "domain": domain,
                             "n_items": len(result["items"]), "mock": mock_mode,
                             "from_end": sample_from_end},
                }
                combo_items_by_lang.setdefault(lang, []).extend(result["items"])

        progress_area.empty()
        for key, msg in errors_by_key.items():
            st.error(f"{key}: {msg}")
        if results_by_key:
            st.session_state["dash_results"] = results_by_key
            st.session_state["dash_lang_view"] = next(iter(results_by_key))

        if not mock_mode and auto_repr and repr_eligible_selected and combo_items_by_lang:
            _run_auto_representational(repr_eligible_selected, combo_items_by_lang)

if item_run_clicked:
    if not selected_item_ids:
        st.sidebar.error("Select at least one item.")
    elif not selected_target_labels:
        st.sidebar.error("Select at least one target model.")
    elif not selected_judges:
        st.sidebar.error("Select at least one judge model.")
    elif not item_run_languages:
        st.sidebar.error("Select at least one language for this item.")
    else:
        item_progress = st.empty()
        item_bar = item_progress.progress(0, text="Starting")
        n_item_langs = len(item_run_languages)
        item_results_by_lang: dict[str, dict] = {}
        item_errors_by_lang: dict[str, str] = {}
        # One timestamp for the whole click, shared across every language it runs -- so a
        # single "Run selected item(s)" click groups under one results/item_run/<run_ts>/
        # folder, and a later click gets its own, never overwriting this one.
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        for li, lang in enumerate(item_run_languages):
            # A fresh Config per language, same reasoning as the full-sweep loop above:
            # run_evaluation writes the resolved dataset path back onto cfg.run.dataset.
            run_cfg = load_config()
            run_cfg.run.language = lang
            run_cfg.run.item_ids = list(selected_item_ids)
            run_cfg.run.behaviours = []
            run_cfg.run.domains = []
            run_cfg.run.limit_per_behaviour = 0
            run_cfg.run.limit_total = 0
            run_cfg.run.target_model_ids = list(selected_target_labels)
            run_cfg.judges.models = list(selected_judges)
            run_cfg.judges.provider = judge_provider
            # A separate, timestamped results tree so a quick one-item test never overwrites,
            # or gets mixed into, a full sweep's results/<lang>[/<domain>] output -- and so
            # successive item runs stay individually reloadable instead of clobbering each other.
            run_cfg.run.output_dir = f"results/item_run/{run_ts}/{lang}"

            def _item_tick(frac: float, msg: str, _i=li) -> None:
                item_bar.progress(min(max((_i + frac) / n_item_langs, 0.0), 1.0),
                                  text=f"[{lang}] {msg} ({_i + 1}/{n_item_langs})")

            try:
                result = run_evaluation(run_cfg, mock=mock_mode, progress=_item_tick)
            except RuntimeError as e:
                item_errors_by_lang[lang] = f"{e}\n\nTurn on Mock mode in the sidebar to run without API keys."
            except Exception as e:  # noqa: BLE001 -- surfaced as a message, not a traceback
                item_errors_by_lang[lang] = f"Run failed: {e}"
            else:
                item_results_by_lang[f"{run_ts}__{lang}"] = {
                    "summary": result["summary"], "paths": result["paths"],
                    "report_text": result["report_text"],
                    "meta": {"targets": selected_target_labels, "judges": selected_judges,
                             "judge_provider": judge_provider, "language": lang,
                             "run_ts": run_ts, "item_ids": selected_item_ids,
                             "n_items": len(result["items"]), "mock": mock_mode},
                }

        item_progress.empty()
        for lang, msg in item_errors_by_lang.items():
            st.error(f"Item run [{lang}]: {msg}")
        if item_results_by_lang:
            st.session_state["dash_item_results"] = item_results_by_lang
            st.session_state["dash_item_lang_view"] = next(iter(item_results_by_lang))

if load_clicked:
    loaded_results_by_key: dict[str, dict] = {}
    loaded_item_results_by_key: dict[str, dict] = {}
    for key in selected_load_keys:
        summary_path = existing_summaries[key]
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read {summary_path}: {e}")
            continue
        out_dir = summary_path.parent
        report_path = out_dir / "nepsyc_summary_latest.txt"
        paths = {"summary_json": summary_path, "item_scores": out_dir / "item_scores.csv",
                 "raw_responses": out_dir / "raw_responses.csv",
                 "judge_detail": out_dir / "judge_detail.csv", "report_txt": report_path}
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else None

        rel_parts = [] if key == "(root)" else key.split("/")
        if rel_parts and rel_parts[0] == "item_run" and len(rel_parts) == 3:
            # results/item_run/<run_ts>/<language>/ -- route to dash_item_results, keyed the
            # same way a freshly-run item spot-check keys it, so both sit in the same
            # "Viewing" list if a fresh run and a loaded one are both present at once.
            _, run_ts, lang = rel_parts
            loaded_item_results_by_key[f"{run_ts}__{lang}"] = {
                "summary": loaded, "paths": paths, "report_text": report_text,
                # No targets/judges/item_ids/mock recorded -- a loaded summary carries no
                # other run metadata the way a freshly-run item spot-check does.
                "meta": {"targets": None, "judges": None, "judge_provider": None,
                         "language": lang, "run_ts": run_ts, "item_ids": None,
                         "n_items": None, "mock": None},
            }
        else:
            # Infer language/domain from the path segments (e.g. "ne_rom", "en/education")
            # purely so result_label() can render a friendlier "Viewing" option than the bare
            # directory name -- best-effort only, since a loaded summary carries no other run
            # metadata (targets/judges/mock/...) the way a freshly-run sweep does.
            lang = next((p for p in rel_parts if p in LANGUAGES), None)
            domain = next((p for p in rel_parts if p in available_domains), None)
            loaded_results_by_key[key] = {
                "summary": loaded, "paths": paths, "report_text": report_text,
                "meta": {"targets": None, "judges": None, "judge_provider": None,
                         "language": lang, "domain": domain, "n_items": None, "mock": None,
                         "from_end": None},
            }
    # Each bucket is only touched if this load actually included that kind of entry, so
    # loading just a sweep (or just an item run) doesn't wipe out whatever the other section
    # already had in session state from an earlier run or an earlier load.
    if loaded_results_by_key:
        st.session_state["dash_results"] = loaded_results_by_key
        st.session_state["dash_lang_view"] = next(iter(loaded_results_by_key))
    if loaded_item_results_by_key:
        st.session_state["dash_item_results"] = loaded_item_results_by_key
        st.session_state["dash_item_lang_view"] = next(iter(loaded_item_results_by_key))

if competence_run_clicked:
    if not selected_target_labels:
        st.sidebar.error("Select at least one target model.")
    else:
        comp_cfg = load_config()
        comp_cfg.run.target_model_ids = list(selected_target_labels)
        try:
            comp_result = run_competence_sweep(comp_cfg, mock=mock_mode)
        except RuntimeError as e:
            st.error(f"Language competence check failed: {e}\n\n"
                     "Turn on Mock mode in the sidebar to run without API keys.")
        except Exception as e:  # noqa: BLE001 -- surfaced as a message, not a traceback
            st.error(f"Language competence check failed: {e}")
        else:
            st.session_state["dash_competence"] = {
                "summary": comp_result["summary"], "paths": comp_result["paths"],
                "meta": {"targets": selected_target_labels, "mock": mock_mode},
            }

if competence_load_clicked:
    loaded_summary_df = _read_csv(existing_competence_summary)
    if loaded_summary_df is None:
        st.error(f"Could not read {existing_competence_summary}")
    else:
        st.session_state["dash_competence"] = {
            "summary": loaded_summary_df.to_dict("records"),
            "paths": {"competence_scores": competence_dir / "competence_scores.csv",
                      "competence_summary": existing_competence_summary},
            "meta": {"targets": None, "mock": None},
        }

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="ns-masthead"><div class="ns-eyebrow">Sycophancy benchmark</div>'
    '<h1>NepSyc</h1><div class="ns-sub">Six behaviours, three language splits, '
    'scored by an LLM judge panel.</div></div>',
    unsafe_allow_html=True,
)

comp_state = st.session_state.get("dash_competence")
if comp_state:
    _section("Language competence", "Verdict by model")
    comp_summary_rows = comp_state["summary"]
    overall_by_model = {r["model"]: r for r in comp_summary_rows if _s(r.get("direction")) == "overall"}
    comp_models = sorted(overall_by_model)
    if not comp_models:
        st.caption("This check produced no scored models.")
    else:
        badges = [_competence_badge_html(m, _s(overall_by_model[m].get("verdict")),
                                         overall_by_model[m].get("bleu"), overall_by_model[m].get("chrfpp"))
                  for m in comp_models]
        st.markdown(f'<div class="ns-strip">{"".join(badges)}</div>', unsafe_allow_html=True)

        for m in comp_models:
            with st.expander(f"{m}  ·  breakdown by direction"):
                rows = [r for r in comp_summary_rows if r["model"] == m]
                bdf = (pd.DataFrame(rows).set_index("direction")[["n_items", "bleu", "chrfpp"]]
                      .rename(columns={"n_items": "items", "bleu": "BLEU", "chrfpp": "chrF++"}))
                st.dataframe(bdf, width="stretch")

        comp_paths = {k: Path(v) for k, v in comp_state["paths"].items() if v}
        cdl = st.columns(2)
        _dl(cdl[0], "competence_scores.csv", comp_paths.get("competence_scores"))
        _dl(cdl[1], "competence_summary.csv", comp_paths.get("competence_summary"))

# ---------------------------------------------------------------------------
# Representational learning -- moved to its own page (Representational Learning, in the
# sidebar nav). Still a pure reader of data/representation/metrics/ and
# results/representations/; nothing on this page loads model weights or triggers extraction.
# ---------------------------------------------------------------------------

_section(
    "Representational learning", "Moved to its own page",
    "Sycophantic vs. neutral hidden states, layer by layer, plus the optional research "
    "panels, now live on the Representational Learning page (sidebar). It renders "
    "independently of any run here -- open it any time.",
)

# ---------------------------------------------------------------------------
# Item spot-check -- results from the sidebar's "Spot-check a single item" panel. Its own
# session-state key and its own results/item_run/<run_ts>/<lang> tree (a fresh timestamp per
# run), kept separate from a full sweep's dash_results so a quick one-item test never gets
# mixed into, or overwrites, real sweep results shown further down.
# ---------------------------------------------------------------------------

item_dash_results = st.session_state.get("dash_item_results")
if item_dash_results:
    _section(
        "Item spot-check", "Run history for a hand-picked item id",
        "From the sidebar's \"Spot-check a single item\" panel -- the exact prompts, "
        "replies, and score(s) for whichever item id(s) and language(s) you picked.",
    )
    item_lang_keys = list(item_dash_results)
    if len(item_lang_keys) > 1:
        default_item_view = st.session_state.get("dash_item_lang_view", item_lang_keys[0])
        if default_item_view not in item_lang_keys:
            default_item_view = item_lang_keys[0]
        active_item_lang = st.radio(
            "Viewing", item_lang_keys, index=item_lang_keys.index(default_item_view),
            horizontal=True, key="dash_item_lang_view",
            format_func=lambda k: result_label(item_dash_results[k]["meta"], k),
        )
    else:
        active_item_lang = item_lang_keys[0]
    item_state = item_dash_results[active_item_lang]
    item_paths = {k: (Path(v) if v else None) for k, v in item_state["paths"].items()}
    item_scores_df = _read_csv(item_paths.get("item_scores"))
    item_raw_df = _read_csv(item_paths.get("raw_responses"))
    item_judge_df = _read_csv(item_paths.get("judge_detail"))

    if item_scores_df is None or item_scores_df.empty:
        st.caption("No scored rows for this item run.")
    else:
        item_preview_of: dict = {}
        if item_raw_df is not None and {"model", "item_id", "turn"} <= set(item_raw_df.columns):
            item_preview_of = item_raw_df.groupby(["model", "item_id"], sort=False)["turn"].first().to_dict()

        item_scores_df = item_scores_df.sort_values(["item_id", "model"]).reset_index(drop=True)

        def _item_label(r) -> str:
            metric = METRIC_OF.get(r.behaviour, r.behaviour)
            score_s = _fmt(r.score, signed=metric in SIGNED_METRICS) if r.score == r.score else "n/a"
            preview = _preview_snippet(item_preview_of.get((r.model, r.item_id)))
            tail = f"  ·  {preview}" if preview else ""
            return f"{r.item_id}  ·  {metric} {score_s}  ·  {r.model}{tail}"

        item_labels = [_item_label(r) for r in item_scores_df.itertuples()]
        item_pick = st.selectbox("Item x model", item_labels, key="dash_item_pick")
        item_row = item_scores_df.iloc[item_labels.index(item_pick)]

        item_detail_raw = item_row.get("detail_json")
        item_detail = {}
        if isinstance(item_detail_raw, str) and item_detail_raw:
            try:
                item_detail = json.loads(item_detail_raw)
            except json.JSONDecodeError:
                item_detail = {}

        item_metric = METRIC_OF.get(item_row["behaviour"], item_row["behaviour"])
        item_score_val = item_row.get("score")
        item_score_val = None if item_score_val != item_score_val else item_score_val
        st.markdown(
            _hero_html(item_score_val, item_metric, item_row["behaviour"], item_detail.get("errors")),
            unsafe_allow_html=True,
        )
        st.markdown(_badges_html(item_row["behaviour"], item_detail), unsafe_allow_html=True)
        st.caption(f'{item_row["item_id"]}  ·  seed {item_row.get("seed_id")}  ·  '
                  f'topic {item_row.get("topic")}  ·  source {item_row.get("source")}')

        if item_raw_df is not None:
            item_turns = item_raw_df[(item_raw_df["model"] == item_row["model"])
                                     & (item_raw_df["item_id"] == item_row["item_id"])]
            if not item_turns.empty:
                st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                            'Prompts and replies</div>', unsafe_allow_html=True)
                st.markdown(_conversation_html(item_turns), unsafe_allow_html=True)

        if item_judge_df is not None:
            item_votes = item_judge_df[(item_judge_df["model"] == item_row["model"])
                                       & (item_judge_df["item_id"] == item_row["item_id"])]
            if not item_votes.empty:
                st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                            'Judge panel</div>', unsafe_allow_html=True)
                st.markdown(_judge_cards_html(item_votes), unsafe_allow_html=True)
                if "prompt" in item_votes.columns:
                    with st.expander("Grading prompt sent to the judge panel"):
                        for call in item_votes["call"].unique():
                            p = item_votes.loc[item_votes["call"] == call, "prompt"].iloc[0]
                            if isinstance(p, str) and p:
                                st.markdown(f"**{call}**")
                                st.text(p)
    st.divider()

dash_results = st.session_state.get("dash_results")
if not dash_results:
    _section("No run loaded", "Start a sweep or open the last one",
             "Configure the run in the sidebar. Mock mode needs no API key and finishes in "
             "seconds. If results/summary.json already exists, Load last results renders it "
             "without spending any calls.")
else:
    lang_keys = list(dash_results)
    if len(lang_keys) > 1:
        default_view = st.session_state.get("dash_lang_view", lang_keys[0])
        if default_view not in lang_keys:
            default_view = lang_keys[0]
        active_lang = st.radio(
            "Viewing", lang_keys, index=lang_keys.index(default_view), horizontal=True,
            key="dash_lang_view",
            format_func=lambda k: result_label(dash_results[k]["meta"], k),
        )
    else:
        active_lang = lang_keys[0]
    result_state = dash_results[active_lang]

    summary = result_state["summary"]
    paths = {k: (Path(v) if v else None) for k, v in result_state["paths"].items()}
    meta = result_state["meta"]

    models = sorted({k.split("||")[0] for k in summary})
    behaviours_present = [b for b in BEHAVIOURS if any(f"{m}||{b}" in summary for m in models)]
    raw_df = _read_csv(paths.get("raw_responses"))
    judge_df = _read_csv(paths.get("judge_detail"))

    # First prompt sent for each (model, item_id), for the "Scored item" picker below --
    # groupby().first() keeps each group's first row in raw_responses.csv's own order,
    # which is that item's first condition and turn, written in whatever language this
    # run's dataset was in, not a fixed English label.
    preview_of: dict = {}
    if raw_df is not None and {"model", "item_id", "turn"} <= set(raw_df.columns):
        preview_of = raw_df.groupby(["model", "item_id"], sort=False)["turn"].first().to_dict()

    if not models:
        st.warning("This run produced no scored results.")
    else:
        color_by_model = color_map(models)
        cov = _coverage(summary, models, behaviours_present)
        coll = _collection_health(raw_df) if raw_df is not None and "reply" in raw_df.columns else None
        if judge_df is not None and "judge_model" in judge_df.columns:
            per_judge, panel = _judge_health(judge_df)
        else:
            per_judge = None
            panel = {"judges_per_call": None, "single": 0, "total": 0,
                     "spread": None, "unanimous": None}

        if meta["targets"] is not None:
            st.markdown(
                f'<div class="ns-eyebrow" style="margin-top:14px;">'
                f'{html.escape(meta["language"])} &nbsp;/&nbsp; {html.escape(meta.get("domain") or "")} '
                f'&nbsp;/&nbsp; {meta["n_items"]} items '
                f'{"(last N per behaviour) " if meta.get("from_end") else ""}'
                f'&nbsp;/&nbsp; judges via {html.escape(meta["judge_provider"])} '
                f'&nbsp;/&nbsp; {"mock" if meta["mock"] else "live"}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                        'Loaded from results on disk</div>', unsafe_allow_html=True)

        _section("Status", "Run integrity",
                 "These four gauges decide whether anything further down is worth reading.")

        blank_rate = float(coll["blank_rate"].max()) if coll is not None and not coll.empty else None
        judge_err = None
        if per_judge is not None and not per_judge.empty and per_judge["calls"].sum():
            judge_err = float(per_judge["errors"].sum() / per_judge["calls"].sum())
        jpc = panel.get("judges_per_call")

        gauges = [
            _gauge("Coverage", _pct(cov["overall"], 1),
                   f"{cov['scored']} of {cov['items']} items scored",
                   cov["overall"], _status_color(cov["overall"])),
            _gauge("Replies received",
                   _pct(1 - blank_rate, 1) if blank_rate is not None else "n/a",
                   "worst model, non blank turns" if blank_rate is not None else "no raw responses",
                   (1 - blank_rate) if blank_rate is not None else None,
                   _status_color((1 - blank_rate) if blank_rate is not None else None,
                                 ok=1 - BLANK_RATE_BAD, bad=0.8)),
            _gauge("Judges per call", _fmt(jpc, 2) if jpc is not None else "n/a",
                   f"{panel['single']} calls got one judge" if panel.get("total") else "no judge detail",
                   (jpc / 3.0) if jpc is not None else None,
                   _status_color((jpc / 3.0) if jpc is not None else None, ok=0.95, bad=0.7)),
            _gauge("Judge success", _pct(1 - judge_err, 1) if judge_err is not None else "n/a",
                   "calls returning a usable vote" if judge_err is not None else "no judge detail",
                   (1 - judge_err) if judge_err is not None else None,
                   _status_color((1 - judge_err) if judge_err is not None else None,
                                 ok=1 - JUDGE_ERROR_BAD, bad=0.75)),
        ]
        st.markdown(f'<div class="ns-strip">{"".join(gauges)}</div>', unsafe_allow_html=True)

        _section("Reading", "What this run supports",
                 "Generated from the aggregates below. Claims marked Observed rest on non "
                 "overlapping 95% intervals. Everything else is a constraint on what can be "
                 "concluded yet.")
        _render_reading(_reading(summary, models, behaviours_present, cov, coll, per_judge, panel))

        _section(
            "Per-behaviour scoring", "Moved to its own page",
            "Headline scores, coverage, collection/judging health, and the per-behaviour "
            "charts (95% bootstrap intervals) now live on the Scoring page (sidebar) -- it "
            "reads this same run, nothing to re-run.",
        )

        _section("Prompt inspector", "Walk one behaviour down to a single prompt",
                 "Pick a behaviour, then a model, then one scored item: the exact prompts sent, "
                 "the model's replies, the score they earned, and the judge panel's rationale.")
        scores_df = _read_csv(paths.get("item_scores"))
        if scores_df is None:
            st.caption("No item_scores.csv for this run.")
        else:
            f1, f2, f3 = st.columns([2, 2, 1])
            with f1:
                behaviour_filter = st.multiselect(
                    "Behaviour", [b for b in BEHAVIOURS if b in set(scores_df["behaviour"])],
                    default=[b for b in BEHAVIOURS if b in set(scores_df["behaviour"])],
                    format_func=lambda b: f"{METRIC_OF[b]}  {b.replace('_', ' ').title()}",
                )
            with f2:
                model_filter = st.multiselect("Model", sorted(scores_df["model"].dropna().unique()),
                                              default=sorted(scores_df["model"].dropna().unique()))
            with f3:
                unscored_only = st.checkbox("Unscored only", value=False,
                                            help="Items dropped before aggregation.")

            filtered = scores_df[scores_df["model"].isin(model_filter)
                                 & scores_df["behaviour"].isin(behaviour_filter)]
            if unscored_only and "score" in filtered.columns:
                filtered = filtered[filtered["score"].isna()]
            filtered = filtered.sort_values(["behaviour", "model", "item_id"]).reset_index(drop=True)

            if filtered.empty:
                st.caption("Nothing matches these filters.")
            else:
                def _label(r) -> str:
                    metric = METRIC_OF.get(r.behaviour, r.behaviour)
                    score_s = _fmt(r.score, signed=metric in SIGNED_METRICS) if r.score == r.score else "n/a"
                    preview = _preview_snippet(preview_of.get((r.model, r.item_id)))
                    tail = f"  ·  {preview}" if preview else ""
                    return f"{metric} {score_s}  ·  {r.model}  ·  {r.item_id}{tail}"

                labels = [_label(r) for r in filtered.itertuples()]
                pick = st.selectbox("Scored item", labels)
                row = filtered.iloc[labels.index(pick)]

                detail_raw = row.get("detail_json")
                detail = {}
                if isinstance(detail_raw, str) and detail_raw:
                    try:
                        detail = json.loads(detail_raw)
                    except json.JSONDecodeError:
                        detail = {}

                metric = METRIC_OF.get(row["behaviour"], row["behaviour"])
                score_val = row.get("score")
                score_val = None if score_val != score_val else score_val
                st.markdown(
                    _hero_html(score_val, metric, row["behaviour"], detail.get("errors")),
                    unsafe_allow_html=True,
                )
                st.markdown(_badges_html(row["behaviour"], detail), unsafe_allow_html=True)
                st.caption(f'{row["item_id"]}  ·  seed {row.get("seed_id")}  ·  '
                          f'topic {row.get("topic")}  ·  source {row.get("source")}')

                if raw_df is not None:
                    turns = raw_df[(raw_df["model"] == row["model"])
                                   & (raw_df["item_id"] == row["item_id"])]
                    if not turns.empty:
                        st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                                    'Prompts and replies</div>', unsafe_allow_html=True)
                        st.markdown(_conversation_html(turns), unsafe_allow_html=True)

                if judge_df is not None:
                    votes = judge_df[(judge_df["model"] == row["model"])
                                     & (judge_df["item_id"] == row["item_id"])]
                    if not votes.empty:
                        st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                                    'Judge panel</div>', unsafe_allow_html=True)
                        st.markdown(_judge_cards_html(votes), unsafe_allow_html=True)
                        if "prompt" in votes.columns:
                            with st.expander("Grading prompt sent to the judge panel"):
                                for call in votes["call"].unique():
                                    p = votes.loc[votes["call"] == call, "prompt"].iloc[0]
                                    if isinstance(p, str) and p:
                                        st.markdown(f"**{call}**")
                                        st.text(p)

                lang_for_hs = active_lang if active_lang in LANGUAGES else None
                hs_meta = _load_hidden_meta()
                item_hs_meta = None
                if hs_meta is not None and not hs_meta.empty:
                    item_hs_meta = hs_meta[hs_meta["item_id"] == row["item_id"]]
                    if lang_for_hs:
                        item_hs_meta = item_hs_meta[item_hs_meta["language"] == lang_for_hs]

                st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                            'Local hidden-state analysis</div>', unsafe_allow_html=True)
                ineligible = [t.label for t in cfg.target_models if not t.hf_repo_id]
                if item_hs_meta is None or item_hs_meta.empty:
                    st.caption(
                        "No local hidden-state run for this item yet. From a terminal: "
                        f"python scripts/analyze_hidden_states.py --item-id {row['item_id']} "
                        f"--language {lang_for_hs or cfg.run.language}"
                    )
                    if ineligible:
                        st.caption("Not eligible for this feature (no hf_repo_id in config.yaml): "
                                  + ", ".join(ineligible))
                else:
                    hs_replies = _load_hidden_replies(item_hs_meta.iloc[0]["language"])
                    hs_replies_item = (hs_replies[hs_replies["item_id"] == row["item_id"]]
                                       if hs_replies is not None else None)

                    all_hs_labels = sorted(item_hs_meta["model_label"].unique())
                    hs_models = st.multiselect(
                        "Models to show", all_hs_labels, default=all_hs_labels,
                        key=f"hs_models_{row['item_id']}",
                        help="Only target_models with hf_repo_id set in config.yaml can run "
                             "here (weights loaded locally) -- that excludes any API-only "
                             "model such as OpenAI's.",
                    )
                    if ineligible:
                        st.caption("Not eligible (no local weights configured): " + ", ".join(ineligible))

                    if not hs_models:
                        st.caption("Select at least one model.")
                    elif hs_replies_item is None or hs_replies_item.empty:
                        st.caption("meta.csv references a run whose replies.csv is missing.")
                    else:
                        conditions_present = sorted(hs_replies_item["condition"].unique())
                        hsc1, hsc2 = st.columns([2, 1])
                        with hsc1:
                            hs_condition = st.selectbox(
                                "Condition", conditions_present,
                                format_func=lambda c: COND_LABELS.get(c, c.replace("_", " ").title()),
                                key=f"hs_cond_{row['item_id']}",
                            )
                        turns_present = sorted(
                            hs_replies_item.loc[hs_replies_item["condition"] == hs_condition, "turn_index"].unique()
                        )
                        with hsc2:
                            hs_turn = (st.selectbox("Turn", turns_present, format_func=lambda i: f"Turn {i + 1}",
                                                    key=f"hs_turn_{row['item_id']}")
                                      if len(turns_present) > 1 else turns_present[0])

                        vecs_by_model = {}
                        for m in hs_models:
                            mrow = item_hs_meta[item_hs_meta["model_label"] == m]
                            if mrow.empty:
                                continue
                            vdf = _read_csv(ROOT / mrow.iloc[0]["vectors_path"])
                            if vdf is not None:
                                vecs_by_model[m] = vdf
                        hs_color_by_model = color_map(sorted(vecs_by_model))

                        st.markdown('<div class="ns-eyebrow" style="margin-top:10px;">'
                                    'Local replies</div>', unsafe_allow_html=True)
                        for m in hs_models:
                            turns = hs_replies_item[(hs_replies_item["model"] == m)
                                                    & (hs_replies_item["condition"] == hs_condition)]
                            if not turns.empty:
                                with st.expander(f"{m}  ·  reply"):
                                    st.markdown(_conversation_html(turns), unsafe_allow_html=True)

                        mag_fig = _hs_magnitude_fig(vecs_by_model, hs_condition, hs_turn, hs_color_by_model)
                        if mag_fig is not None:
                            st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                                        'Layer magnitude (L2 norm)</div>', unsafe_allow_html=True)
                            st.plotly_chart(mag_fig, width="stretch")

                        pca_fig = _hs_pca_fig(vecs_by_model, hs_condition, hs_turn, hs_color_by_model)
                        if pca_fig is not None:
                            st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                                        'Layer trajectory (PCA of the full hidden vector)</div>',
                                        unsafe_allow_html=True)
                            st.plotly_chart(pca_fig, width="stretch")

                        st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                                    'Per-dimension heatmap</div>', unsafe_allow_html=True)
                        for m in hs_models:
                            if m not in vecs_by_model:
                                continue
                            heat_fig = _hs_heatmap_fig(vecs_by_model[m], hs_condition, hs_turn)
                            if heat_fig is not None:
                                with st.expander(f"{m}  ·  all hidden dimensions by layer",
                                                 expanded=len(hs_models) == 1):
                                    st.plotly_chart(heat_fig, width="stretch")

                        st.caption(
                            "prompt_type is 'sycophantic' for every row here -- these conditions are "
                            "themselves the sycophancy-framed prompts NepSyc builds. A normal/unframed "
                            "counterpart isn't in the dataset yet; once build_dataset.py adds one, this "
                            "same vectors CSV schema (prompt_type column) is ready for a cosine-"
                            "similarity comparison without a format change."
                        )

                with st.expander("Raw scoring detail (detail_json)"):
                    if detail:
                        st.json(detail)
                    else:
                        st.caption("detail_json could not be parsed.")

        _section("Files", "Everything this run wrote")
        dl = st.columns(5)

        _dl(dl[0], "summary.json", paths.get("summary_json"), "application/json")
        _dl(dl[1], "item_scores.csv", paths.get("item_scores"))
        _dl(dl[2], "raw_responses.csv", paths.get("raw_responses"))
        _dl(dl[3], "judge_detail.csv", paths.get("judge_detail"))
        report_text = result_state.get("report_text")
        if report_text:
            dl[4].download_button("summary.txt", data=report_text,
                                  file_name="nepsyc_summary_latest.txt",
                                  mime="text/plain", width="stretch")
        else:
            dl[4].caption("report: n/a")

        with st.expander("Full text report"):
            st.text(report_text) if report_text else st.caption("Not available.")
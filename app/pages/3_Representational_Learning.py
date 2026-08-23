"""Representational learning: sycophantic vs. neutral hidden states, layer by layer, plus
five optional research panels -- pulled off the main NepSyc page so these charts get their
own place instead of sitting at the bottom of one long page.

A third standalone axis alongside Language Competence and the Prompt Inspector's own "Local
hidden-state analysis" sub-block (both still on the main page): extraction and metric
aggregation both run offline (scripts/extract_representations.py loads real multi-GB model
weights; scripts/analyze_representation_drift.py and scripts/analyze_representation_research.py
are pure numpy/pandas readers of that output), so this page only ever reads
data/representation/metrics/ and results/representations/ already on disk -- nothing here
loads a model or triggers a run.

    streamlit run app/dashboard.py   (this page is reachable from its sidebar nav)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dash_common import (  # noqa: E402
    CATEGORICAL, COND_LABELS, CSS, LANGUAGE_LABELS, _fmt, _read_csv,
    badges_html as _badges_html, hero_html as _hero_html, section,
)
from nepsyc.config import load_config  # noqa: E402
from nepsyc.metrics import METRIC_OF  # noqa: E402

st.set_page_config(page_title="NepSyc · Representational Learning", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="ns-masthead"><div class="ns-eyebrow">Sycophancy benchmark</div>'
    '<h1>Representational learning</h1><div class="ns-sub">Sycophantic vs. neutral hidden '
    'states, layer by layer, open-weight models only.</div></div>',
    unsafe_allow_html=True,
)

cfg = load_config()

REPR_METRICS_DIR = ROOT / "data" / "representation" / "metrics"
REPR_RESULTS_DIR = ROOT / "results" / "representations"
REPR_POOLINGS = ["last_token", "mean_pooled"]
REPR_POOLING_LABELS = {"last_token": "Last token (pre-answer state)", "mean_pooled": "Mean-pooled (own reply)"}


@st.cache_data(show_spinner=False)
def _load_repr_tidy() -> pd.DataFrame | None:
    return _read_csv(REPR_METRICS_DIR / "layer_cosine.csv")


@st.cache_data(show_spinner=False)
def _load_repr_layer_agg() -> pd.DataFrame | None:
    return _read_csv(REPR_METRICS_DIR / "layer_agg.csv")


@st.cache_data(show_spinner=False)
def _load_repr_index() -> pd.DataFrame | None:
    return _read_csv(REPR_RESULTS_DIR / "index.csv")


@st.cache_data(show_spinner=False)
def _load_repr_item_scores(language: str) -> pd.DataFrame | None:
    """Sycophancy-sweep item_scores.csv for this language, if one has been run -- checked
    under both the multi-language (results/<lang>/) and single-language (results/) layouts,
    same precedence the rest of the dashboard uses. Representation extraction and the main
    sycophancy sweep are run independently, so this can legitimately come back empty."""
    for p in (ROOT / "results" / language / "item_scores.csv", ROOT / "results" / "item_scores.csv"):
        df = _read_csv(p)
        if df is not None:
            return df
    return None


@st.cache_data(show_spinner=False)
def _load_repr_meta(meta_path: str) -> dict | None:
    p = ROOT / meta_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def _load_repr_tensors(tensors_path: str) -> dict[str, np.ndarray] | None:
    """Loads exactly one pair/variant's tensors.npz -- a few KB, not the whole extraction
    store -- only for whatever the filters below currently have selected."""
    p = ROOT / tensors_path
    if not p.exists():
        return None
    with np.load(p) as npz:
        return {k: npz[k] for k in npz.files}


def _repr_layer_line_fig(sub: pd.DataFrame, y_col: str, y_title: str) -> go.Figure | None:
    """One trace per pooling variant, layer on the x-axis -- the primary "how similar is this
    prompt pair's representation, layer by layer" chart."""
    fig = go.Figure()
    plotted = False
    for i, pooling in enumerate(REPR_POOLINGS):
        s = sub[sub["pooling"] == pooling].sort_values("layer")
        if s.empty:
            continue
        plotted = True
        fig.add_scatter(
            x=s["layer"], y=s[y_col], mode="lines+markers", name=REPR_POOLING_LABELS[pooling],
            line=dict(color=CATEGORICAL[i * 3 % len(CATEGORICAL)], width=2), marker=dict(size=6),
            hovertemplate=f"{pooling}<br>layer %{{x}}<br>{y_title} %{{y:.4f}}<extra></extra>",
        )
    if not plotted:
        return None
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=300,
        xaxis_title="layer (0 = embedding)", yaxis_title=y_title,
        legend=dict(orientation="h", y=-0.22),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _repr_model_layer_heatmap(agg: pd.DataFrame) -> go.Figure | None:
    """Model x layer heatmap of mean cosine distance, already scoped to the caller's current
    behaviour/domain/language/pooling filter -- shows exactly the filtered set, not the
    unfiltered global average model_layer_matrix_*.csv on disk carries."""
    if agg.empty:
        return None
    pivot = (agg.pivot_table(index="model", columns="layer", values="mean_cosine_distance", aggfunc="mean")
             .sort_index(axis=1))
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues",
        colorbar=dict(title="mean cos. dist.", thickness=12),
        hovertemplate="model %{y}<br>layer %{x}<br>mean cosine distance %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=max(160, 56 * len(pivot.index) + 70),
        xaxis_title="layer", font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    # Plotly heatmaps plot y[0] at the bottom by default -- reverse so the first row (here,
    # alphabetically first model) reads at the top, top-to-bottom like every other list here.
    fig.update_yaxes(autorange="reversed")
    return fig


def _repr_all_prompts_heatmap(sub: pd.DataFrame) -> go.Figure | None:
    """pair(y) x layer(x) heatmap of every prompt this model has a representation run for,
    ordered behaviour -> domain -> language -> pair_id -> condition -> turn -- the "every
    prompt's layer profile at once" survey view, independent of the behaviour/domain/language
    filters that scope the panels above/below it."""
    if sub.empty:
        return None
    sub = sub.copy()
    sub["row_label"] = (
        sub["behaviour"].str.replace("_", " ") + " · " + sub["domain"] + " · " + sub["language"]
        + " · " + sub["pair_id"] + " · " + sub["condition"] + " t" + sub["turn_index"].astype(str)
    )
    order_cols = ["behaviour", "domain", "language", "pair_id", "condition", "turn_index"]
    row_order = (sub[["row_label"] + order_cols].drop_duplicates()
                 .sort_values(order_cols, kind="stable")["row_label"].tolist())
    pivot = sub.pivot_table(index="row_label", columns="layer", values="cosine_distance", aggfunc="mean")
    pivot = pivot.reindex(row_order).sort_index(axis=1)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Blues",
        colorbar=dict(title="cos. dist.", thickness=12),
        hovertemplate="%{y}<br>layer %{x}<br>cosine distance %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=min(1400, max(240, 20 * len(pivot.index) + 80)),
        xaxis_title="layer", font=dict(family="IBM Plex Mono, monospace", size=10),
    )
    # Plotly heatmaps plot y[0] at the bottom by default -- reverse so row_order (behaviour ->
    # domain -> language -> pair_id -> condition -> turn) reads top-to-bottom as intended.
    fig.update_yaxes(autorange="reversed")
    return fig


def _repr_norm_fig(layers: list, syco_norms: list, neutral_norms: list) -> go.Figure:
    """Per-layer activation-norm bars, sycophantic vs. neutral variant of the same prompt --
    activations/hidden states from a live forward pass, not the model's static weights."""
    fig = go.Figure()
    fig.add_bar(x=layers, y=syco_norms, name="sycophantic variant", marker_color=CATEGORICAL[0])
    fig.add_bar(x=layers, y=neutral_norms, name="neutral variant", marker_color=CATEGORICAL[4])
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=300,
        xaxis_title="layer (0 = embedding)", yaxis_title="activation L2 norm",
        legend=dict(orientation="h", y=-0.22),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _repr_activation_norms(tensors: dict, turn_index: int, pooling: str, n_layers: int) -> list:
    norms = []
    for layer in range(n_layers):
        vec = tensors.get(f"t{turn_index}_L{layer}_{pooling}")
        norms.append(float(np.linalg.norm(vec.astype(np.float64))) if vec is not None else None)
    return norms


def _repr_metric_badges_html(row: dict, mean_dist, last_layer_dist) -> str:
    chips = []
    for label, val, signed in (
        ("mean cosine distance", mean_dist, False),
        ("cosine distance, last layer", last_layer_dist, False),
        ("confidence shift (P correct)", row.get("confidence_shift"), True),
        ("logit-preference shift Δ", row.get("logit_preference_shift_delta"), True),
    ):
        if val is None or (isinstance(val, float) and val != val):
            continue
        chips.append(f'<div class="ns-badge"><span class="ns-badge-k">{html.escape(label)}</span>'
                     f'{html.escape(_fmt(val, 4, signed=signed))}</div>')
    return f'<div class="ns-badgerow">{"".join(chips)}</div>' if chips else ""


def _repr_variant_card(title: str, prompt, reply) -> str:
    prompt_html = html.escape(prompt or "")
    reply_html = html.escape(reply or "")
    return (
        f'<div class="ns-condlabel">{html.escape(title)}</div>'
        f'<div class="ns-chat">'
        f'<div class="ns-bubble ns-user"><div class="ns-role">Prompt</div>'
        f'<div class="ns-text">{prompt_html}</div></div>'
        f'<div class="ns-bubble ns-assistant"><div class="ns-role">Model reply</div>'
        f'<div class="ns-text">{reply_html}</div></div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Optional research panels (scripts/analyze_representation_research.py). Five independent,
# collapsible panels, each a pure reader of its own research_*.csv/.npz -- nothing here
# re-derives anything scripts/analyze_representation_research.py hasn't already precomputed
# offline. Every panel is gated behind the same repr_model/repr_pooling picked by the core
# section's filters above, so switching model/pooling there re-scopes every research panel too.
# ---------------------------------------------------------------------------

REPR_VARIANT_COLORS = {"sycophantic": CATEGORICAL[0], "neutral": CATEGORICAL[4]}


@st.cache_data(show_spinner=False)
def _load_repr_research(name: str) -> pd.DataFrame | None:
    return _read_csv(REPR_METRICS_DIR / f"research_{name}.csv")


def _repr_pca_scatter_fig(sub: pd.DataFrame) -> go.Figure | None:
    if sub.empty:
        return None
    fig = go.Figure()
    for variant in ("sycophantic", "neutral"):
        s = sub[sub["variant"] == variant]
        if s.empty:
            continue
        fig.add_scatter(
            x=s["pca_x"], y=s["pca_y"], mode="markers", name=variant,
            marker=dict(size=9, color=REPR_VARIANT_COLORS[variant],
                       line=dict(width=1, color="rgba(255,255,255,0.6)")),
            text=s["behaviour"] + " · " + s["pair_id"] + " · " + s["condition"] + " t" + s["turn_index"].astype(str),
            hovertemplate="%{text}<br>PC1 %{x:.3f}  PC2 %{y:.3f}<extra>%{fullData.name}</extra>",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=380,
        xaxis_title="PC1", yaxis_title="PC2", legend=dict(orientation="h", y=-0.16),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _repr_line_by_group_fig(sub: pd.DataFrame, group_col: str, y_col: str, y_title: str,
                            highlight: str | None = None) -> go.Figure | None:
    """One trace per distinct value of group_col, layer on the x-axis -- shared shape for the
    direction-norm, CKA, and cross-behaviour-stability charts below. `highlight`, if given
    (e.g. "__all__"), is drawn thicker/solid while every other trace is drawn thin/dashed, so
    the one scope with real statistical power doesn't visually disappear next to five
    low-n behaviour lines."""
    if sub.empty:
        return None
    fig = go.Figure()
    groups = sorted(sub[group_col].unique(), key=lambda g: (g != highlight, str(g)))
    for i, g in enumerate(groups):
        s = sub[sub[group_col] == g].sort_values("layer")
        is_main = g == highlight
        fig.add_scatter(
            x=s["layer"], y=s[y_col], mode="lines+markers", name=str(g).replace("_", " "),
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=3 if is_main else 1.5,
                      dash=None if is_main else "dot"),
            marker=dict(size=5 if is_main else 3),
            hovertemplate=f"{g}<br>layer %{{x}}<br>{y_title} %{{y:.4f}}<extra></extra>",
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=320,
        xaxis_title="layer (0 = embedding)", yaxis_title=y_title,
        legend=dict(orientation="h", y=-0.28),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _repr_rup_trajectory_fig(sub: pd.DataFrame) -> go.Figure | None:
    if sub.empty:
        return None
    s = sub.sort_values("turn_index")
    fig = go.Figure()
    fig.add_scatter(x=s["turn_index"], y=s["cumulative_drift"], mode="lines+markers",
                    name="cumulative step drift (turn-to-turn)", line=dict(color=CATEGORICAL[0], width=3))
    fig.add_scatter(x=s["turn_index"], y=s["cosine_distance_from_neutral"], mode="lines+markers",
                    name="cosine distance from neutral", line=dict(color=CATEGORICAL[4], width=2, dash="dot"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=300,
        xaxis_title="pressure turn", yaxis_title="cosine distance",
        legend=dict(orientation="h", y=-0.24),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1, tickvals=s["turn_index"], ticktext=[f"turn {t+1}" for t in s["turn_index"]])
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    return fig


def _repr_fertility_fig(sub: pd.DataFrame) -> go.Figure | None:
    if sub.empty:
        return None
    s = sub.sort_values("layer")
    fig = go.Figure()
    fig.add_scatter(x=s["layer"], y=s["spearman_rho"], mode="lines+markers", name="Spearman ρ",
                    line=dict(color=CATEGORICAL[0], width=2.5))
    fig.add_scatter(x=s["layer"], y=s["r_squared"], mode="lines+markers", name="linear R²",
                    line=dict(color=CATEGORICAL[3], width=2, dash="dash"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=300,
        xaxis_title="layer (0 = embedding)", yaxis_title="correlation (fertility vs. cosine distance)",
        legend=dict(orientation="h", y=-0.24),
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, dtick=1)
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False, range=[-1, 1])
    return fig


# ---------------------------------------------------------------------------
# Page -- own section, renders unconditionally (no dependency on dash_results/dash_lang_view,
# unlike the Scoring page). A pure reader of whatever scripts/extract_representations.py +
# scripts/analyze_representation_drift.py have already written to disk; no run button here
# since extraction loads real model weights and can take minutes on CPU (same reason the main
# page's Prompt Inspector hidden-state sub-block has no button).
# ---------------------------------------------------------------------------

st.markdown(
    section(
        "Representational learning", "Sycophantic vs. neutral hidden states, layer by layer",
        "Open-weight models only (target_models with hf_repo_id set in config.yaml). Reads "
        "precomputed cosine-similarity metrics and per-turn tensors/replies already written by "
        "scripts/extract_representations.py and scripts/analyze_representation_drift.py -- "
        "nothing here re-extracts or loads model weights.",
    ),
    unsafe_allow_html=True,
)

repr_open_labels = [t.label for t in cfg.target_models if t.hf_repo_id]
repr_tidy = _load_repr_tidy()
repr_agg = _load_repr_layer_agg()
repr_index = _load_repr_index()
# results/representations/ (raw tensors.npz + index.csv) is gitignored, so a fresh clone
# or machine can have the committed data/representation/metrics/ aggregates -- everything
# the model/behaviour/domain/pair filters and the per-layer/heatmap charts below need --
# with no local index.csv at all. Only gate the whole page on repr_tidy; repr_index is
# checked again, separately, right before the one section (sycophantic-vs-neutral tensors)
# that actually needs it.
if repr_tidy is None or repr_tidy.empty:
    st.caption(
        "No representation-drift metrics yet. From a terminal: "
        "python scripts/extract_representations.py --model <label> --behaviour <behaviour> "
        "--limit 2   (writes results/representations/), then "
        "python scripts/analyze_representation_drift.py   (writes data/representation/metrics/)."
    )
    if repr_open_labels:
        st.caption("Eligible (hf_repo_id set in config.yaml): " + ", ".join(repr_open_labels))
    else:
        st.caption("No target_models in config.yaml have hf_repo_id set yet -- nothing is eligible.")
else:
    if repr_index is None:
        repr_index = pd.DataFrame(columns=["model_label", "pair_id", "variant", "turn_index",
                                            "n_layers", "tensors_path", "meta_path"])
    if repr_open_labels:
        repr_tidy = repr_tidy[repr_tidy["model"].isin(repr_open_labels)]
        repr_index = repr_index[repr_index["model_label"].isin(repr_open_labels)]
    if repr_tidy.empty:
        st.caption("The models in data/representation/metrics/ don't match any current "
                  "open-weight target_models -- config.yaml may have changed since extraction.")
    else:
        st.markdown('<div class="ns-eyebrow" style="margin-top:6px;">Filters</div>', unsafe_allow_html=True)
        rf1, rf2, rf3, rf4 = st.columns(4)
        with rf1:
            repr_model = st.selectbox("Model", sorted(repr_tidy["model"].unique()), key="repr_model")
        model_sub = repr_tidy[repr_tidy["model"] == repr_model]
        with rf2:
            repr_behaviour = st.selectbox(
                "Behaviour", sorted(model_sub["behaviour"].unique()),
                format_func=lambda b: b.replace("_", " ").title(), key="repr_behaviour",
            )
        behaviour_sub = model_sub[model_sub["behaviour"] == repr_behaviour]
        with rf3:
            repr_domain = st.selectbox("Domain", sorted(behaviour_sub["domain"].unique()), key="repr_domain")
        domain_sub = behaviour_sub[behaviour_sub["domain"] == repr_domain]
        with rf4:
            repr_language = st.selectbox(
                "Language", sorted(domain_sub["language"].unique()),
                format_func=lambda l: LANGUAGE_LABELS.get(l, l), key="repr_language",
            )
        language_sub = domain_sub[domain_sub["language"] == repr_language]

        pair_options = sorted(language_sub["pair_id"].unique())
        rf5, rf6, rf7, rf8 = st.columns(4)
        with rf5:
            repr_pair = st.selectbox("Prompt pair", pair_options, key="repr_pair")
        pair_sub = language_sub[language_sub["pair_id"] == repr_pair]
        cond_options = sorted(pair_sub["condition"].unique())
        with rf6:
            repr_condition = st.selectbox(
                "Condition (sycophantic side)", cond_options,
                format_func=lambda c: COND_LABELS.get(c, c.replace("_", " ").title()), key="repr_condition",
            )
        cond_sub = pair_sub[pair_sub["condition"] == repr_condition]
        turn_options = sorted(cond_sub["turn_index"].unique())
        with rf7:
            repr_turn = (st.selectbox("Turn", turn_options, format_func=lambda t: f"Turn {t + 1}",
                                      key="repr_turn") if len(turn_options) > 1 else turn_options[0])
        with rf8:
            repr_pooling = st.radio(
                "Pooling", REPR_POOLINGS, format_func=lambda p: REPR_POOLING_LABELS[p],
                horizontal=True, key="repr_pooling",
            )

        sel = cond_sub[cond_sub["turn_index"] == repr_turn]
        sel_pool = sel[sel["pooling"] == repr_pooling].sort_values("layer")

        # --- 2. Cosine-similarity graphs ------------------------------------
        st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                    'Per-layer cosine similarity — selected pair</div>', unsafe_allow_html=True)
        line_fig = _repr_layer_line_fig(sel, "cosine_similarity", "cosine similarity")
        if line_fig is not None:
            st.plotly_chart(line_fig, width="stretch", key="repr_line_fig")
        else:
            st.caption("No cosine data for this exact pair/condition/turn.")

        st.markdown(
            '<div class="ns-eyebrow" style="margin-top:16px;">'
            f'Model × layer heatmap — {repr_behaviour.replace("_", " ")} · {repr_domain} · '
            f'{LANGUAGE_LABELS.get(repr_language, repr_language)}</div>', unsafe_allow_html=True,
        )
        agg_filtered = pd.DataFrame()
        if repr_agg is not None and not repr_agg.empty:
            agg_filtered = repr_agg[
                (repr_agg["behaviour"] == repr_behaviour) & (repr_agg["domain"] == repr_domain)
                & (repr_agg["language"] == repr_language) & (repr_agg["pooling"] == repr_pooling)
            ]
            if repr_open_labels:
                agg_filtered = agg_filtered[agg_filtered["model"].isin(repr_open_labels)]
        heat_fig = _repr_model_layer_heatmap(agg_filtered)
        if heat_fig is not None:
            st.plotly_chart(heat_fig, width="stretch", key="repr_model_layer_heatmap")
        else:
            st.caption("No layer_agg.csv rows for this filter set.")

        survey_sub = repr_tidy[(repr_tidy["model"] == repr_model) & (repr_tidy["pooling"] == repr_pooling)]
        n_survey_pairs = survey_sub["pair_id"].nunique() if not survey_sub.empty else 0
        st.markdown(
            '<div class="ns-eyebrow" style="margin-top:16px;">'
            f'All prompts for {repr_model} — ordered behaviour · domain · language · pair'
            '</div>', unsafe_allow_html=True,
        )
        with st.expander(f"{n_survey_pairs} prompt pair(s) — click to expand the full survey heatmap"):
            survey_fig = _repr_all_prompts_heatmap(survey_sub)
            if survey_fig is not None:
                st.plotly_chart(survey_fig, width="stretch", key="repr_survey_heatmap")
            else:
                st.caption("No rows for this model.")

        # --- 3. Sycophantic vs. neutral representation comparison ----------
        st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                    'Sycophantic vs. neutral — internal representations</div>', unsafe_allow_html=True)
        neutral_variant = sel_pool["neutral_variant"].iloc[0] if not sel_pool.empty else None
        idx_syco = repr_index[
            (repr_index["model_label"] == repr_model) & (repr_index["pair_id"] == repr_pair)
            & (repr_index["variant"] == repr_condition) & (repr_index["turn_index"] == repr_turn)
        ]
        idx_neutral = (repr_index[
            (repr_index["model_label"] == repr_model) & (repr_index["pair_id"] == repr_pair)
            & (repr_index["variant"] == neutral_variant) & (repr_index["turn_index"] == 0)
        ] if neutral_variant else pd.DataFrame())

        meta_syco = _load_repr_meta(idx_syco.iloc[0]["meta_path"]) if not idx_syco.empty else None
        meta_neutral = _load_repr_meta(idx_neutral.iloc[0]["meta_path"]) if not idx_neutral.empty else None

        if repr_index.empty:
            st.caption(
                "No results/representations/index.csv on this machine -- it's gitignored "
                "(the raw tensors.npz it points at are large binaries), so a fresh clone "
                "has the committed data/representation/metrics/ aggregates above but not "
                "the per-prompt tensors this section reads. Re-run "
                "`python scripts/extract_representations.py --model <label> --behaviour "
                f"{repr_behaviour} --limit 2` locally to populate it."
            )
        elif idx_syco.empty or idx_neutral.empty:
            st.caption("Missing tensors.npz for the sycophantic or neutral side of this pair — "
                      "re-run scripts/extract_representations.py for this item.")
        else:
            tensors_syco = _load_repr_tensors(idx_syco.iloc[0]["tensors_path"])
            tensors_neutral = _load_repr_tensors(idx_neutral.iloc[0]["tensors_path"])
            n_layers = int(idx_syco.iloc[0]["n_layers"])
            if tensors_syco is None or tensors_neutral is None:
                st.caption("tensors.npz referenced by the index is missing on disk.")
            else:
                syco_norms = _repr_activation_norms(tensors_syco, repr_turn, repr_pooling, n_layers)
                neutral_norms = _repr_activation_norms(tensors_neutral, 0, repr_pooling, n_layers)
                st.caption(
                    "Activations / hidden states from a live forward pass over each prompt — not "
                    "the model's static weights. A large per-layer distance between the two curves "
                    "below means the model's internal state for the sycophancy-framed prompt has "
                    "diverged substantially from its neutral twin by that layer: the framing has "
                    "measurably changed how the model represents the question internally, not just "
                    "what it eventually says."
                )
                st.plotly_chart(_repr_norm_fig(list(range(n_layers)), syco_norms, neutral_norms),
                                width="stretch", key="repr_norm_fig")
                cos_fig = _repr_layer_line_fig(sel, "cosine_similarity", "cosine similarity")
                if cos_fig is not None:
                    st.plotly_chart(cos_fig, width="stretch", key="repr_cos_fig")

        # --- 4. One-place comparison panel ----------------------------------
        st.markdown('<div class="ns-eyebrow" style="margin-top:16px;">'
                    'Prompt → output → judge score → internal drift</div>', unsafe_allow_html=True)
        st.caption(f'{repr_pair}  ·  {repr_behaviour.replace("_", " ").title()}  ·  {repr_domain}  ·  '
                  f'{LANGUAGE_LABELS.get(repr_language, repr_language)}')

        syco_turn_meta = None
        if meta_syco:
            syco_turn_meta = next((t for t in meta_syco.get("turns", []) if t.get("turn_index") == repr_turn), None)
        neutral_turn_meta = meta_neutral["turns"][0] if meta_neutral and meta_neutral.get("turns") else None

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(
                _repr_variant_card(
                    f"Sycophantic · {COND_LABELS.get(repr_condition, repr_condition)}",
                    syco_turn_meta.get("prompt") if syco_turn_meta else None,
                    syco_turn_meta.get("reply") if syco_turn_meta else None,
                ),
                unsafe_allow_html=True,
            )
        with cc2:
            st.markdown(
                _repr_variant_card(
                    f"Neutral · {COND_LABELS.get(neutral_variant, neutral_variant or 'n/a')}",
                    neutral_turn_meta.get("prompt") if neutral_turn_meta else None,
                    neutral_turn_meta.get("reply") if neutral_turn_meta else None,
                ),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                    'Judge score — sycophantic variant</div>', unsafe_allow_html=True)
        scores_df_repr = _load_repr_item_scores(repr_language)
        judge_row = None
        if scores_df_repr is not None:
            m = scores_df_repr[(scores_df_repr["model"] == repr_model) & (scores_df_repr["item_id"] == repr_pair)]
            if not m.empty:
                judge_row = m.iloc[0]
        if judge_row is None:
            st.caption(f"No judge score for {repr_model} on {repr_pair} — the sycophancy sweep "
                      "(`python run.py evaluate`) hasn't scored this model/item yet.")
        else:
            jdetail = {}
            jdetail_raw = judge_row.get("detail_json")
            if isinstance(jdetail_raw, str) and jdetail_raw:
                try:
                    jdetail = json.loads(jdetail_raw)
                except json.JSONDecodeError:
                    jdetail = {}
            jmetric = METRIC_OF.get(judge_row["behaviour"], judge_row["behaviour"])
            jscore = judge_row.get("score")
            jscore = None if jscore != jscore else jscore
            st.markdown(_hero_html(jscore, jmetric, judge_row["behaviour"], jdetail.get("errors"), compact=True),
                        unsafe_allow_html=True)
            st.markdown(_badges_html(judge_row["behaviour"], jdetail), unsafe_allow_html=True)

        st.markdown('<div class="ns-eyebrow" style="margin-top:14px;">'
                    'Representation metrics</div>', unsafe_allow_html=True)
        if sel_pool.empty:
            st.caption("No representation metrics for this exact selection.")
        else:
            mean_dist = float(sel_pool["cosine_distance"].mean())
            last_layer_dist = float(sel_pool.iloc[-1]["cosine_distance"])
            st.markdown(_repr_metric_badges_html(sel_pool.iloc[0].to_dict(), mean_dist, last_layer_dist),
                        unsafe_allow_html=True)

        # --- 5. Optional research panels --------------------------------
        st.markdown('<div class="ns-eyebrow" style="margin-top:20px;">'
                    'Research panels (optional)</div>', unsafe_allow_html=True)
        st.caption(
            "Five additional, independent panels from scripts/analyze_representation_research.py -- "
            "each reads its own precomputed research_*.csv, nothing here re-derives anything live. "
            "Scoped to the model/pooling picked above. Sample sizes are small throughout "
            f"({repr_model}: {repr_tidy[repr_tidy['model'] == repr_model]['pair_id'].nunique()} "
            "item-pair(s) total) -- every panel's caption states its own n and how to read it."
        )

        pca_research = _load_repr_research("pca_layers")
        direction_research = _load_repr_research("directions")
        stability_research = _load_repr_research("direction_stability")
        cka_research = _load_repr_research("cka")
        rup_research = _load_repr_research("rup_drift")
        fertility_research = _load_repr_research("fertility")
        research_missing = all(
            d is None or d.empty for d in
            (pca_research, direction_research, cka_research, rup_research, fertility_research)
        )
        if research_missing:
            st.caption(
                "No research-panel data yet. From a terminal: "
                "python scripts/analyze_representation_research.py   "
                "(reuses results/representations/ and data/representation/metrics/layer_cosine.csv, "
                "no model weights loaded except a lightweight tokenizer for the fertility panel)."
            )
        else:
            with st.expander("PCA projection at a chosen layer, coloured by variant", expanded=False):
                st.caption(
                    "2D PCA fit independently at each layer over every point available at that "
                    "layer for this model/pooling: one point per sycophantic condition/turn, plus "
                    "one neutral point per item (deduplicated across the conditions that share it). "
                    "This is a per-layer fit, not a shared cross-layer basis, matching the Prompt "
                    "Inspector's own per-turn PCA convention -- PC1/PC2 at layer 5 are not the same "
                    "axes as at layer 20. Caveat: fit over roughly 20-40 points depending on layer; "
                    "a 2D projection with this few points shows gross separation at best, not fine "
                    "cluster structure. Layer 0 (the raw embedding of the fixed chat-template "
                    "suffix token) is expected to collapse to a single point -- the model hasn't "
                    "seen the actual prompt content yet at that position."
                )
                pf = pca_research[
                    (pca_research["model"] == repr_model) & (pca_research["pooling"] == repr_pooling)
                ] if pca_research is not None else pd.DataFrame()
                if pf.empty:
                    st.caption("No PCA fit for this model/pooling.")
                else:
                    pca_layer_options = sorted(pf["layer"].unique())
                    pca_layer = st.select_slider("Layer", options=pca_layer_options,
                                                 value=pca_layer_options[len(pca_layer_options) // 2],
                                                 key="repr_pca_layer")
                    layer_sub = pf[pf["layer"] == pca_layer]
                    fig = _repr_pca_scatter_fig(layer_sub)
                    if fig is not None:
                        evr = layer_sub.iloc[0]
                        st.caption(f"n={int(evr['n_points'])} points · explained variance "
                                  f"PC1={evr['explained_variance_ratio_1']:.1%} "
                                  f"PC2={evr['explained_variance_ratio_2']:.1%}")
                        st.plotly_chart(fig, width="stretch", key="repr_pca_fig")

            with st.expander("Sycophancy direction (difference-in-means): norm + cross-behaviour stability",
                             expanded=False):
                st.caption(
                    "DIRECTION = mean, over matched item/condition/turn pairs, of "
                    "(sycophantic-condition vector − neutral vector) -- the same paired "
                    "difference-of-means construction representation-engineering steering vectors "
                    "use (e.g. contrastive activation addition). This is a DERIVED DIRECTION, not a "
                    "trained probe weight -- no classifier was fit, only an average taken. Solid "
                    "line is __all__ (every behaviour pooled); dotted lines are individual "
                    "behaviours. Caveat: attribution_bias contributes only 1 item-pair, so its line "
                    "(and any cross-behaviour cosine involving it) is one item's own difference, not "
                    "a behaviour-general estimate -- read the norm chart's per-behaviour lines as "
                    "illustrative, and lean on __all__ for anything resembling a claim."
                )
                df_ = direction_research[
                    (direction_research["model"] == repr_model) & (direction_research["pooling"] == repr_pooling)
                ] if direction_research is not None else pd.DataFrame()
                fig = _repr_line_by_group_fig(df_, "behaviour", "direction_norm",
                                              "‖direction‖", highlight="__all__")
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key="repr_direction_fig")
                else:
                    st.caption("No direction data for this model/pooling.")

                st.markdown('<div class="ns-eyebrow" style="margin-top:10px;">'
                            'Cross-behaviour cosine stability</div>', unsafe_allow_html=True)
                sf = stability_research[
                    (stability_research["model"] == repr_model) & (stability_research["pooling"] == repr_pooling)
                ] if stability_research is not None else pd.DataFrame()
                if sf.empty:
                    st.caption("Fewer than two behaviours have a direction at this model/pooling.")
                else:
                    sf = sf.copy()
                    sf["pair_label"] = sf["behaviour_a"].str.replace("_", " ") + " × " + sf["behaviour_b"].str.replace("_", " ")
                    mean_stability = sf.groupby("layer", as_index=False)["cosine"].mean().rename(columns={"cosine": "mean_cosine"})
                    fig2 = go.Figure()
                    fig2.add_scatter(x=mean_stability["layer"], y=mean_stability["mean_cosine"],
                                     mode="lines+markers", name="mean pairwise cosine",
                                     line=dict(color=CATEGORICAL[0], width=3))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=8, r=8, t=8, b=8), height=260,
                        xaxis_title="layer (0 = embedding)", yaxis_title="mean cosine between behaviour directions",
                        font=dict(family="IBM Plex Mono, monospace", size=11),
                    )
                    fig2.update_xaxes(showgrid=False, dtick=1)
                    fig2.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False, range=[-1, 1])
                    st.plotly_chart(fig2, width="stretch", key="repr_stability_fig")
                    st.caption(
                        "Averaged across every behaviour-pair at that layer -- high mean cosine means "
                        "different behaviours' sycophancy directions point the same way (one shared "
                        "'sycophancy axis'); low/negative means they diverge. Most behaviour-pairs "
                        "here are flagged low_n (fewer than 3 matched pairs on at least one side, "
                        "the same MIN_CKA_N threshold scripts/analyze_representation_research.py "
                        "uses) -- treat this curve's shape as suggestive, not confirmed."
                    )

            with st.expander("Linear CKA per layer: sycophantic vs. neutral", expanded=False):
                st.caption(
                    "Linear Centered Kernel Alignment (Kornblith et al. 2019) between the "
                    "sycophantic-side and neutral-side representation matrices at each layer, via "
                    "the Gram-matrix/HSIC formulation. 1.0 = the two conditions' representation "
                    "geometries are identical up to rotation/scale; 0 = unrelated. Solid line is "
                    "__all__ (every matched pair pooled, the only scope with real n); dotted lines "
                    "are individual behaviours, shown only where that behaviour has at least "
                    "3 matched pairs at that layer -- fewer than that isn't computed at all, not "
                    "silently rounded. Caveat: even __all__'s n (roughly 15-20 pairs) is a sketch, "
                    "not a converged CKA estimate; read differences between layers as directional, "
                    "not precise. NaN at layer 0 is expected (see the PCA panel's note on the "
                    "embedding layer collapsing to a single point)."
                )
                cf = cka_research[
                    (cka_research["model"] == repr_model) & (cka_research["pooling"] == repr_pooling)
                ] if cka_research is not None else pd.DataFrame()
                fig = _repr_line_by_group_fig(cf, "scope", "cka", "linear CKA", highlight="__all__")
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key="repr_cka_fig")
                else:
                    st.caption("No CKA data for this model/pooling.")

            with st.expander("Revision-under-pressure: drift trajectory across turns", expanded=False):
                st.caption(
                    "revision_under_pressure only. Two curves per pressure item: (1) cumulative "
                    "step drift -- the running sum of turn-to-turn cosine distance between the "
                    "pressure condition's OWN hidden states (turn 1 vs. turn 0, turn 2 vs. turn 1), "
                    "independent of the neutral anchor -- total path length the internal state has "
                    "travelled by that turn; (2) cosine distance from neutral at each turn, reusing "
                    "the core section's own already-computed metric for context. A widening gap "
                    "between the two suggests the state is moving in a direction that keeps taking "
                    "it further from its neutral starting point rather than wandering back toward "
                    "it. Caveat: only 2 revision_under_pressure item-pairs exist -- read each "
                    "trajectory as one item's own path, not a population trend."
                )
                rf = rup_research[
                    (rup_research["model"] == repr_model) & (rup_research["pooling"] == repr_pooling)
                ] if rup_research is not None else pd.DataFrame()
                if rf.empty:
                    st.caption("No revision_under_pressure drift data for this model/pooling.")
                else:
                    rup_pairs = sorted(rf["pair_id"].unique())
                    rup_pair = st.selectbox("Item pair", rup_pairs, key="repr_rup_pair")
                    rup_layers = sorted(rf[rf["pair_id"] == rup_pair]["layer"].unique())
                    rup_layer = st.select_slider("Layer", options=rup_layers,
                                                 value=rup_layers[len(rup_layers) // 2], key="repr_rup_layer")
                    trajectory_sub = rf[(rf["pair_id"] == rup_pair) & (rf["layer"] == rup_layer)]
                    fig = _repr_rup_trajectory_fig(trajectory_sub)
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch", key="repr_rup_fig")

            with st.expander("Tokenizer fertility vs. per-layer drift", expanded=False):
                st.caption(
                    "Spearman ρ and linear R² between each prompt's tokenizer fertility (tokens per "
                    "whitespace word -- higher means the tokenizer fragments this text into more "
                    "subword pieces) and its cosine distance from the neutral twin, correlated "
                    "across every matched item/condition/turn available at that layer. Fertility "
                    "itself doesn't depend on layer; what varies is whether a heavily-fragmented "
                    "prompt's representation tends to drift further at some layers than others. "
                    "Caveat: only 1 model has been extracted so far, so 'per model' is n=1 model "
                    "today -- no cross-model pattern can be read yet. Within that model, n is "
                    "roughly 20-40 points per layer; a ρ around 0.3-0.5 with p > 0.05 (common in "
                    "this data) should be read as a weak, not-yet-significant trend, not a finding."
                )
                ff = fertility_research[
                    fertility_research["model"] == repr_model
                ] if fertility_research is not None else pd.DataFrame()
                ff = ff[ff["pooling"] == repr_pooling] if not ff.empty else ff
                fig = _repr_fertility_fig(ff)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch", key="repr_fertility_fig")
                else:
                    st.caption("No fertility-correlation data for this model/pooling.")

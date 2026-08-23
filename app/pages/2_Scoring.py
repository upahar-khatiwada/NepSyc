"""Per-behaviour scoring: headline scores table, coverage, collection/judging health, and one
Plotly bar chart per behaviour with 95% bootstrap intervals -- pulled off the main NepSyc page
so six charts aren't buried under everything else on one long page.

Pure reader of st.session_state["dash_results"] -- it never runs a sweep itself.

    streamlit run app/dashboard.py   (this page is reachable from its sidebar nav)
"""
from __future__ import annotations

import html
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dash_common import (  # noqa: E402
    BLANK_RATE_BAD, CSS, MIN_N_FOR_CI, SIGNED_METRICS, _fmt, _read_csv,
    collection_health as _collection_health, color_map, coverage_summary as _coverage,
    entry as _entry, judge_health as _judge_health, meter_html as _meter, result_label,
    section, status_color as _status_color,
)
from nepsyc.metrics import BEHAVIOURS, METRIC_OF  # noqa: E402
from nepsyc.report import DIRECTION  # noqa: E402

st.set_page_config(page_title="NepSyc · Scoring", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="ns-masthead"><div class="ns-eyebrow">Sycophancy benchmark</div>'
    '<h1>Per-behaviour scoring</h1><div class="ns-sub">Headline scores, coverage, collection '
    'and judging health, and one chart per behaviour with 95% bootstrap intervals.</div></div>',
    unsafe_allow_html=True,
)

dash_results = st.session_state.get("dash_results")
if not dash_results:
    st.markdown(
        section("No run loaded", "Run a sweep or load results first",
                "Go to the NepSyc page (sidebar) and either run a sweep or click "
                "\"Load last results\". This page reads whatever it finds there -- "
                "it does not run anything itself."),
        unsafe_allow_html=True,
    )
    st.stop()

lang_keys = list(dash_results)
active_lang = st.session_state.get("dash_lang_view", lang_keys[0])
if active_lang not in lang_keys:
    active_lang = lang_keys[0]
if len(lang_keys) > 1:
    active_lang = st.radio(
        "Viewing", lang_keys, index=lang_keys.index(active_lang), horizontal=True,
        format_func=lambda k: result_label(dash_results[k]["meta"], k),
    )

result_state = dash_results[active_lang]
summary = result_state["summary"]
paths = {k: (Path(v) if v else None) for k, v in result_state["paths"].items()}
meta = result_state["meta"]

models = sorted({k.split("||")[0] for k in summary})
behaviours_present = [b for b in BEHAVIOURS if any(f"{m}||{b}" in summary for m in models)]

if not models:
    st.warning("This run produced no scored results.")
    st.stop()

raw_df = _read_csv(paths.get("raw_responses"))
judge_df = _read_csv(paths.get("judge_detail"))

color_by_model = color_map(models)
cov = _coverage(summary, models, behaviours_present)
coll = _collection_health(raw_df) if raw_df is not None and "reply" in raw_df.columns else None
if judge_df is not None and "judge_model" in judge_df.columns:
    per_judge, panel = _judge_health(judge_df)
else:
    per_judge = None
    panel = {"judges_per_call": None, "single": 0, "total": 0, "spread": None, "unanimous": None}
blank_rate = float(coll["blank_rate"].max()) if coll is not None and not coll.empty else None

if meta.get("targets") is not None:
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


# ---------------------------------------------------------------------------
# Tables and charts -- page-local, only ever used here (dashboard.py no longer needs them
# after this section moved off the main page). _meter/_status_color/_coverage/
# _collection_health/_judge_health, which both this page and dashboard.py's Status/Reading
# sections need, live in app/dash_common.py instead -- see the import block above.
# ---------------------------------------------------------------------------

BEHAVIOUR_EXTRA_COLS = {
    "revision_under_pressure": ["baseline_accuracy", "flip_rate", "stable_correct_rate", "recovery_rate"],
    "attribution_bias": ["mean_rating_delta", "mean_error_flag_gap"],
    "mirroring": ["mean_abs", "pct_positive"],
    "agreement_bias": ["hard_agreement_rate_mcq"],
}


def _scores_table(summary, models, behaviours, color_by_model) -> str:
    head = "".join(f"<th>{METRIC_OF[b]}</th>" for b in behaviours)
    body = []
    for m in models:
        cells = []
        for b in behaviours:
            e = _entry(summary, m, b)
            if not e or e.get("mean") is None:
                cells.append('<td class="ns-thin"><div class="ns-num">n/a</div>'
                             '<div class="ns-sub2">not scored</div></td>')
                continue
            n = e.get("n_scored") or 0
            ci = e.get("ci95") or [float("nan"), float("nan")]
            thin = ' class="ns-thin"' if n < MIN_N_FOR_CI else ""
            cells.append(
                f'<td{thin}><div class="ns-num">'
                f'{_fmt(e["mean"], signed=METRIC_OF[b] in SIGNED_METRICS)}</div>'
                f'<div class="ns-sub2">n={n} &nbsp; [{_fmt(ci[0])}, {_fmt(ci[1])}]</div></td>'
            )
        body.append(
            f'<tr><td class="ns-rowhead">'
            f'<span class="ns-swatch" style="background:{color_by_model[m]};"></span>'
            f'{html.escape(m)}</td>' + "".join(cells) + "</tr>"
        )
    return (f'<table class="ns-table"><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _coverage_table(cov, models, behaviours) -> str:
    head = "".join(f"<th>{METRIC_OF[b]}</th>" for b in behaviours)
    body = []
    for m in models:
        cells = []
        for b in behaviours:
            cell = cov["cells"].get((m, b))
            if not cell:
                cells.append('<td class="ns-thin"><div class="ns-num">n/a</div></td>')
                continue
            scored, total, frac = cell
            color = _status_color(frac)
            cells.append(f'<td><div class="ns-num" style="color:{color};">{scored}/{total}</div>'
                         f'{_meter(frac, color)}</td>')
        body.append(f'<tr><td class="ns-rowhead">{html.escape(m)}</td>' + "".join(cells) + "</tr>")
    return (f'<table class="ns-table"><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _behaviour_chart(behaviour, summary, models, color_by_model) -> go.Figure | None:
    metric = METRIC_OF[behaviour]
    rows = []
    for m in models:
        e = _entry(summary, m, behaviour)
        if not e or e.get("mean") is None:
            continue
        ci = e.get("ci95") or [float("nan"), float("nan")]
        mean = e["mean"]
        up = max((ci[1] - mean) if ci[1] == ci[1] else 0.0, 0.0)
        dn = max((mean - ci[0]) if ci[0] == ci[0] else 0.0, 0.0)
        rows.append((m, mean, up, dn, e.get("n_scored"), e.get("n_items")))
    if not rows:
        return None

    fig = go.Figure()
    fig.add_bar(
        x=[f"{r[0]}<br><sub>n={r[4]}</sub>" for r in rows],
        y=[r[1] for r in rows],
        error_y=dict(type="data", array=[r[2] for r in rows], arrayminus=[r[3] for r in rows],
                     visible=True, thickness=1.2, width=6),
        marker_color=[color_by_model[r[0]] for r in rows],
        width=0.5,
        customdata=[[r[4], r[5]] for r in rows],
        hovertemplate=(f"%{{x}}<br>{metric} = %{{y:.2f}}"
                       "<br>scored %{customdata[0]} of %{customdata[1]}<extra></extra>"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=310, yaxis_title=metric, bargap=0.45,
        font=dict(family="IBM Plex Mono, monospace", size=11),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(family="Archivo, sans-serif", size=12))
    fig.update_yaxes(gridcolor="rgba(128,138,150,0.22)", zeroline=False)
    if metric in SIGNED_METRICS:
        fig.update_yaxes(range=[-5, 5], zeroline=True, zerolinewidth=1.2,
                         zerolinecolor="rgba(128,138,150,0.65)")
    else:
        fig.update_yaxes(range=[0, 5])
    return fig


def _behaviour_extras(behaviour, summary, models) -> pd.DataFrame | None:
    cols = BEHAVIOUR_EXTRA_COLS.get(behaviour)
    if not cols:
        return None
    rows = []
    for m in models:
        e = _entry(summary, m, behaviour)
        if e:
            rows.append({"model": m, **{c: e.get(c) for c in cols}})
    return pd.DataFrame(rows).set_index("model") if rows else None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    section("Scores", "Mean per model and behaviour",
            "Each cell carries its sample size and interval. Cells scored on fewer than "
            f"{MIN_N_FOR_CI} items are dimmed."),
    unsafe_allow_html=True,
)
st.markdown(_scores_table(summary, models, behaviours_present, color_by_model),
            unsafe_allow_html=True)
st.markdown(
    '<div class="ns-hint" style="margin-top:10px;">'
    + " &nbsp;/&nbsp; ".join(f"<b>{k}</b> {html.escape(v)}" for k, v in DIRECTION.items())
    + "</div>", unsafe_allow_html=True,
)

st.markdown(
    section("Coverage", "Items scored against items attempted",
            "Read this beside the scores. A cell scored on 3 of 16 items still prints a "
            "mean and an interval, and the interval gets narrower as the sample shrinks."),
    unsafe_allow_html=True,
)
st.markdown(_coverage_table(cov, models, behaviours_present), unsafe_allow_html=True)

st.markdown(
    section("Collection and judging", "Where items were lost",
            "Blank replies come from the target models, judge errors from the panel. The "
            "two failure modes look identical in the scores and have different fixes."),
    unsafe_allow_html=True,
)
c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="ns-eyebrow">Target models</div>', unsafe_allow_html=True)
    if coll is not None:
        st.dataframe(
            coll.rename(columns={"blank_rate": "blank rate", "median_chars": "median chars"})
                .style.format({"blank rate": "{:.1%}", "median chars": "{:.0f}"}),
            width="stretch",
        )
        if blank_rate is not None and blank_rate > BLANK_RATE_BAD:
            st.caption("Lower requests_per_minute and max_workers in config.yaml, then "
                       "re-run. Blank replies are not cached, so a re-run retries only "
                       "the failures.")
    else:
        st.caption("No raw_responses.csv for this run.")
with c2:
    st.markdown('<div class="ns-eyebrow">Judge panel</div>', unsafe_allow_html=True)
    if per_judge is not None:
        st.dataframe(per_judge.rename(columns={"error_rate": "error rate"})
                     .style.format({"error rate": "{:.1%}"}), width="stretch")
        bits = []
        if panel.get("spread") is not None:
            bits.append(f"mean spread {panel['spread']:.3f}")
        if panel.get("unanimous") is not None:
            bits.append(f"{panel['unanimous']:.1%} unanimous")
        if bits:
            st.caption(" / ".join(bits) + ", over calls that received at least two judges.")
    else:
        st.caption("No judge_detail.csv for this run.")

st.markdown(
    section("Per behaviour", "Distribution and secondary rates",
            "Error bars are 95% bootstrap intervals. Sample size is printed under each bar."),
    unsafe_allow_html=True,
)
for b in behaviours_present:
    fig = _behaviour_chart(b, summary, models, color_by_model)
    if fig is None:
        continue
    with st.expander(f"{METRIC_OF[b]}   {b.replace('_', ' ').title()}", expanded=True):
        st.caption(DIRECTION[METRIC_OF[b]])
        st.plotly_chart(fig, width="stretch")
        extras = _behaviour_extras(b, summary, models)
        if extras is not None:
            st.dataframe(extras, width="stretch")
        if b == "revision_under_pressure":
            st.caption("Flip, stable and recovery rates cover only items answered "
                       "correctly at turn one. Read them beside baseline accuracy.")
        if b == "attribution_bias":
            st.caption("A blank rating delta means no 'Rating: X/10' line was found in "
                       "the reply, not that the model rated both versions equally.")

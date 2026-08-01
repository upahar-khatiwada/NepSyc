"""Streamlit dashboard for NepSyc: pick models, run the benchmark, read the results.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.config import load_config  # noqa: E402
from nepsyc.metrics import BEHAVIOURS, METRIC_OF  # noqa: E402
from nepsyc.pipeline import list_configured_models, run_evaluation  # noqa: E402
from nepsyc.report import DIRECTION  # noqa: E402

st.set_page_config(page_title="NepSyc", layout="wide")

# Fixed categorical order (validated CVD-safe palette), assigned by model identity and
# never by score rank, so a model keeps its colour across every chart on the page.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
               "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

OK_COLOR = "#1b8a5f"
WARN_COLOR = "#b4690e"
BAD_COLOR = "#a62b2b"
MUTED = "rgba(128,138,150,1)"

SIGNED_METRICS = {"MRS", "ATS", "AIS"}
LANGUAGES = ["en", "ne", "ne_rom"]

# A sweep that silently drops most of its items still produces a full table of means
# and confidence intervals. These thresholds drive the coverage matrix and the reading.
COVERAGE_OK = 0.90
COVERAGE_BAD = 0.50
BLANK_RATE_BAD = 0.05
JUDGE_ERROR_BAD = 0.10
MIN_N_FOR_CI = 10
TARGET_CI_HALFWIDTH = 0.5
FLOOR_MEAN = 0.25
FLOOR_SD = 0.50

BEHAVIOUR_EXTRA_COLS = {
    "revision_under_pressure": ["baseline_accuracy", "flip_rate", "stable_correct_rate", "recovery_rate"],
    "attribution_bias": ["mean_rating_delta", "mean_error_flag_gap"],
    "mirroring": ["mean_abs", "pct_positive"],
    "agreement_bias": ["hard_agreement_rate_mcq"],
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp li, .stApp label { font-family: 'Archivo', system-ui, sans-serif; }

.ns-masthead { padding: 4px 0 2px; }
.ns-masthead h1 {
  font-family: 'Archivo', sans-serif; font-size: 26px; font-weight: 700;
  letter-spacing: -0.02em; margin: 0; line-height: 1.1;
}
.ns-masthead .ns-sub { font-size: 13px; color: rgba(128,138,150,1); margin-top: 4px; }

.ns-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 500;
  letter-spacing: 0.16em; text-transform: uppercase; color: rgba(128,138,150,1);
}

.ns-section { margin: 30px 0 12px; padding-bottom: 7px; border-bottom: 1px solid rgba(128,138,150,0.28); }
.ns-section .ns-title { font-size: 15px; font-weight: 600; letter-spacing: -0.01em; margin-top: 3px; }
.ns-section .ns-hint { font-size: 12.5px; color: rgba(128,138,150,1); margin-top: 5px; line-height: 1.45; }
.ns-hint { font-size: 12.5px; color: rgba(128,138,150,1); line-height: 1.5; }

.ns-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 10px; }
.ns-gauge { border: 1px solid rgba(128,138,150,0.28); padding: 11px 13px 10px; }
.ns-gauge .ns-val {
  font-family: 'IBM Plex Mono', monospace; font-size: 21px; font-weight: 500;
  font-variant-numeric: tabular-nums; margin-top: 6px; line-height: 1;
}
.ns-gauge .ns-foot { font-size: 11px; color: rgba(128,138,150,1); margin-top: 7px; }

.ns-meter { height: 3px; background: rgba(128,138,150,0.22); margin-top: 8px; }
.ns-meter > span { display: block; height: 100%; }

.ns-table { width: 100%; border-collapse: collapse; }
.ns-table th {
  font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 500;
  letter-spacing: 0.14em; text-transform: uppercase; color: rgba(128,138,150,1);
  text-align: left; padding: 0 12px 9px; border-bottom: 1px solid rgba(128,138,150,0.32);
}
.ns-table td { padding: 11px 12px; border-bottom: 1px solid rgba(128,138,150,0.16); vertical-align: top; }
.ns-table td.ns-rowhead {
  font-family: 'Archivo', sans-serif; font-size: 13px; font-weight: 500; white-space: nowrap;
}
.ns-table .ns-num {
  font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 500;
  font-variant-numeric: tabular-nums; line-height: 1;
}
.ns-table .ns-sub2 {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px;
  color: rgba(128,138,150,1); margin-top: 5px; line-height: 1.3;
}
.ns-table td.ns-thin { opacity: 0.45; }
.ns-swatch { display: inline-block; width: 8px; height: 8px; margin-right: 8px; vertical-align: 1px; }

.ns-note { display: grid; grid-template-columns: 84px 1fr; gap: 16px; padding: 13px 0; border-bottom: 1px solid rgba(128,138,150,0.16); }
.ns-tag {
  font-family: 'IBM Plex Mono', monospace; font-size: 9px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; padding-top: 5px; border-top: 2px solid;
}
.ns-note .ns-claim { font-size: 13.5px; line-height: 1.5; }
.ns-note .ns-ev {
  display: block; margin-top: 5px; font-family: 'IBM Plex Mono', monospace;
  font-size: 11.5px; color: rgba(128,138,150,1); line-height: 1.5;
}

.ns-hero {
  border: 1px solid rgba(128,138,150,0.28); padding: 16px 20px; margin: 4px 0 4px;
  display: flex; align-items: baseline; gap: 22px; flex-wrap: wrap;
}
.ns-hero .ns-hero-val {
  font-family: 'IBM Plex Mono', monospace; font-size: 38px; font-weight: 600;
  font-variant-numeric: tabular-nums; line-height: 1;
}
.ns-hero .ns-hero-metric {
  font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: rgba(128,138,150,1);
  margin-left: 6px; letter-spacing: 0.06em;
}
.ns-hero .ns-hero-meta { font-size: 12.5px; color: rgba(128,138,150,1); line-height: 1.6; }
.ns-hero .ns-hero-meta b { color: inherit; font-family: 'Archivo', sans-serif; }

.ns-badgerow { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 2px; }
.ns-badge {
  border: 1px solid rgba(128,138,150,0.28); padding: 5px 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; line-height: 1.3;
}
.ns-badge .ns-badge-k { color: rgba(128,138,150,1); margin-right: 6px; }

.ns-condlabel {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: rgba(128,138,150,1);
  margin: 18px 0 8px; padding-top: 12px; border-top: 1px solid rgba(128,138,150,0.2);
}
.ns-condlabel:first-child { margin-top: 6px; padding-top: 0; border-top: none; }

.ns-chat { display: flex; flex-direction: column; gap: 8px; }
.ns-bubble {
  border: 1px solid rgba(128,138,150,0.26); border-left: 3px solid rgba(128,138,150,0.5);
  padding: 10px 14px;
}
.ns-bubble.ns-user { border-left-color: #2a78d6; }
.ns-bubble.ns-assistant { border-left-color: #1baf7a; margin-left: 22px; }
.ns-bubble.ns-error { border-left-color: #a62b2b; margin-left: 22px; }
.ns-bubble .ns-role {
  font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: rgba(128,138,150,1);
  margin-bottom: 6px;
}
.ns-bubble .ns-text { font-size: 13.5px; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }

.ns-jcard {
  border: 1px solid rgba(128,138,150,0.26); padding: 10px 14px; margin-bottom: 8px;
}
.ns-jcard .ns-jhead {
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px;
}
.ns-jcard .ns-jmodel { font-weight: 600; }
.ns-jcard .ns-jval { font-weight: 600; font-variant-numeric: tabular-nums; }
.ns-jcard .ns-jrationale {
  font-size: 12.5px; color: rgba(128,138,150,1); margin-top: 7px; line-height: 1.55;
}
.ns-jcard .ns-jerror { font-size: 12px; color: #a62b2b; margin-top: 7px; }
</style>
"""


def _fmt(v, places=2, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:+.{places}f}" if signed else f"{v:.{places}f}"


def _pct(v, places=0):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v * 100:.{places}f}%"


def _meter(fraction, color) -> str:
    f = 0.0 if fraction is None or fraction != fraction else max(0.0, min(1.0, fraction))
    return f'<div class="ns-meter"><span style="width:{f * 100:.1f}%;background:{color};"></span></div>'


def _status_color(fraction, ok=COVERAGE_OK, bad=COVERAGE_BAD) -> str:
    if fraction is None or fraction != fraction:
        return MUTED
    if fraction < bad:
        return BAD_COLOR
    if fraction < ok:
        return WARN_COLOR
    return OK_COLOR


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


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _entry(summary: dict, model: str, behaviour: str) -> dict | None:
    return summary.get(f"{model}||{behaviour}")


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


def _coverage(summary: dict, models: list[str], behaviours: list[str]) -> dict:
    cells, scored_total, item_total = {}, 0, 0
    for m in models:
        for b in behaviours:
            e = _entry(summary, m, b)
            if not e:
                continue
            total = e.get("n_items") or 0
            scored = e.get("n_scored") or 0
            cells[(m, b)] = (scored, total, (scored / total) if total else None)
            scored_total += scored
            item_total += total
    return {
        "cells": cells,
        "overall": (scored_total / item_total) if item_total else None,
        "scored": scored_total,
        "items": item_total,
        "short": [(m, b, s, t, f) for (m, b), (s, t, f) in cells.items()
                  if f is not None and f < COVERAGE_OK],
    }


def _collection_health(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["len"] = d["reply"].fillna("").astype(str).str.len()
    g = d.groupby("model").agg(turns=("len", "size"), blank=("len", lambda s: int((s == 0).sum())))
    g["blank_rate"] = g["blank"] / g["turns"]
    live = d[d["len"] > 0]
    g["median_chars"] = live.groupby("model")["len"].median() if not live.empty else None
    return g


def _judge_health(jd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_judge = jd.groupby("judge_model").agg(
        calls=("judge_model", "size"),
        errors=("judge_error", "count"),
    )
    per_judge["error_rate"] = per_judge["errors"] / per_judge["calls"]

    numeric = jd.copy()
    numeric["v"] = pd.to_numeric(numeric["judge_value"], errors="coerce")
    numeric = numeric[numeric["v"].notna()]

    panel = {"judges_per_call": None, "single": 0, "total": 0, "spread": None, "unanimous": None}
    if not numeric.empty:
        grp = numeric.groupby(["model", "item_id", "call"])["v"]
        sizes, spread = grp.size(), grp.max() - grp.min()
        multi = spread[sizes >= 2]
        panel["judges_per_call"] = float(sizes.mean())
        panel["single"] = int((sizes < 2).sum())
        panel["total"] = int(len(sizes))
        if not multi.empty:
            panel["spread"] = float(multi.mean())
            panel["unanimous"] = float((multi == 0).mean())
    return per_judge, panel


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
            notes.append(_note(
                "blocked",
                "Empty replies are reaching the judges. A blank reply is scored as a hedge "
                "rather than dropped, which pulls scores toward the middle of the scale.",
                f"{who} returned {worst_rate:.0%} blank turns "
                f"({int(coll.loc[who, 'blank'])} of {int(coll.loc[who, 'turns'])})",
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
# Tables and charts
# ---------------------------------------------------------------------------

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
# Item inspector: prompt, reply and judge verdict for one scored item
# ---------------------------------------------------------------------------

COND_LABELS = {
    "main": "Main", "pressure": "Pressure (3 turns)",
    "stance_pro": "Stance — pro", "stance_con": "Stance — con",
    "self_authored": "Self-authored", "anonymous": "Anonymous",
    "self_opinion": "Self opinion", "authority_cue": "Authority cue",
}

# (detail_json key, label, formatter) shown as badges under the score hero, per behaviour.
DETAIL_FIELDS = {
    "agreement_bias": [("parsed_letter", "MCQ letter"), ("hard_agreed_with_user", "agreed w/ user"),
                        ("hard_correct", "hard correct")],
    "delusion_acceptance": [],
    "revision_under_pressure": [("grading_mode", "grading"), ("eligible", "eligible"),
                                 ("turn_labels", "turn labels"), ("flip", "flipped"),
                                 ("stable_correct", "stable correct"), ("recovery", "recovered")],
    "mirroring": [("stance_pro", "stance pro"), ("stance_con", "stance con"), ("abs_score", "|MRS|")],
    "attribution_bias": [("positivity_self", "positivity self"), ("positivity_anon", "positivity anon"),
                          ("rating_self", "rating self"), ("rating_anon", "rating anon"),
                          ("rating_delta", "rating delta"), ("error_flag_gap", "error-flag gap")],
    "authority_influence": [("agreement_self", "agreement self"), ("agreement_authority", "agreement authority")],
}


def _s(v) -> str:
    """Stringify a cell that may be pandas NaN (read back from a blank CSV field) or None."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _score_color(value, metric) -> str:
    if value is None or value != value:
        return MUTED
    mag = min(abs(value) / 5.0, 1.0)
    if mag < 0.2:
        return OK_COLOR
    if mag < 0.5:
        return WARN_COLOR
    return BAD_COLOR


def _badge_value(v) -> str | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, list):
        return " → ".join(str(x) for x in v)
    return str(v)


def _hero_html(score, metric, behaviour, errors) -> str:
    color = _score_color(score, metric)
    if score is None or score != score:
        val = "n/a"
        meta = "not scored" + (f" — {html.escape('; '.join(errors))}" if errors else "")
    else:
        val = _fmt(score, signed=metric in SIGNED_METRICS)
        meta = html.escape(DIRECTION.get(metric, ""))
    return (
        f'<div class="ns-hero">'
        f'<div><span class="ns-hero-val" style="color:{color};">{html.escape(val)}</span>'
        f'<span class="ns-hero-metric">{html.escape(metric)}</span></div>'
        f'<div class="ns-hero-meta"><b>{html.escape(behaviour.replace("_", " ").title())}</b><br>{meta}</div>'
        f'</div>'
    )


def _badges_html(behaviour, detail: dict) -> str:
    fields = DETAIL_FIELDS.get(behaviour, [])
    chips = []
    for key, label in fields:
        v = _badge_value(detail.get(key))
        if v is None:
            continue
        chips.append(f'<div class="ns-badge"><span class="ns-badge-k">{html.escape(label)}</span>'
                     f'{html.escape(v)}</div>')
    return f'<div class="ns-badgerow">{"".join(chips)}</div>' if chips else ""


def _conversation_html(turns: pd.DataFrame) -> str:
    blocks = []
    for cond in turns["condition"].unique():
        sub = turns[turns["condition"] == cond].sort_values("turn_index")
        blocks.append(f'<div class="ns-condlabel">{html.escape(COND_LABELS.get(cond, cond.replace("_", " ").title()))}</div>')
        bubbles = []
        err = _s(sub.iloc[0].get("error")).strip()
        for _, r in sub.iterrows():
            prompt = _s(r.get("turn"))
            bubbles.append(
                f'<div class="ns-bubble ns-user"><div class="ns-role">User &middot; turn {int(r["turn_index"]) + 1}</div>'
                f'<div class="ns-text">{html.escape(prompt)}</div></div>'
            )
            reply = _s(r.get("reply"))
            if reply:
                bubbles.append(
                    f'<div class="ns-bubble ns-assistant"><div class="ns-role">Model reply</div>'
                    f'<div class="ns-text">{html.escape(reply)}</div></div>'
                )
        if err:
            bubbles.append(f'<div class="ns-bubble ns-error"><div class="ns-role">Error</div>'
                           f'<div class="ns-text">{html.escape(err)}</div></div>')
        blocks.append(f'<div class="ns-chat">{"".join(bubbles)}</div>')
    return "".join(blocks)


def _judge_cards_html(votes: pd.DataFrame) -> str:
    blocks = []
    for call in votes["call"].unique():
        sub = votes[votes["call"] == call]
        blocks.append(f'<div class="ns-condlabel">Judge votes &middot; {html.escape(str(call))}</div>')
        cards = []
        for _, r in sub.iterrows():
            val_s = _s(r.get("judge_value"))
            rationale = _s(r.get("judge_rationale")).strip()
            err = _s(r.get("judge_error")).strip()
            cards.append(
                '<div class="ns-jcard">'
                f'<div class="ns-jhead"><span class="ns-jmodel">{html.escape(_s(r.get("judge_model")))}</span>'
                f'<span class="ns-jval">{html.escape(val_s) if val_s else "n/a"}</span></div>'
                + (f'<div class="ns-jrationale">{html.escape(rationale)}</div>' if rationale else "")
                + (f'<div class="ns-jerror">{html.escape(err)}</div>' if err else "")
                + '</div>'
            )
        blocks.append("".join(cards))
    return "".join(blocks)


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

judge_options = list(info["judges"])
selected_judges = st.sidebar.multiselect(
    "Judge models", judge_options, default=judge_options,
    help="The median vote across these models is taken.",
)
provider_options = sorted(info["providers"])
default_judge_provider = cfg.judges.provider or "groq"
judge_provider = st.sidebar.selectbox(
    "Judge provider", provider_options,
    index=provider_options.index(default_judge_provider) if default_judge_provider in provider_options else 0,
    help="All judge calls in one sweep go through a single provider.",
)

selected_ids = {t["id"] for t in info["targets"] if t["label"] in selected_target_labels}
overlap = sorted(set(selected_judges) & selected_ids)
if overlap:
    st.sidebar.warning(
        f"{', '.join(overlap)} is both a target and a judge. It will grade its own replies, "
        "and on a shared gateway both roles draw down one rate limit."
    )

language = st.sidebar.selectbox("Language", LANGUAGES, index=LANGUAGES.index(cfg.run.language))
selected_behaviours = st.sidebar.multiselect(
    "Behaviours", BEHAVIOURS, default=BEHAVIOURS,
    format_func=lambda b: f"{b.replace('_', ' ').title()} ({METRIC_OF[b]})",
)
items_per_behaviour = st.sidebar.number_input("Items per behaviour", 1, 200, 2, 1)

mock_mode = st.sidebar.toggle("Mock mode", value=True)
st.sidebar.caption(
    "Mock mode is free and instant: every model returns the same canned reply, which is "
    "enough to check wiring. Turn it off for a real sweep, with the providers' keys in .env."
)
st.sidebar.caption(
    f"Throughput from config.yaml: {cfg.run.max_workers} workers at "
    f"{cfg.run.requests_per_minute} requests per minute. Going over a provider's limit "
    "returns blank replies rather than an error, so check coverage after any change."
)

run_clicked = st.sidebar.button("Run benchmark", type="primary", width="stretch")
st.sidebar.divider()
existing_summary = ROOT / "results" / "summary.json"
load_clicked = st.sidebar.button(
    "Load last results", disabled=not existing_summary.exists(), width="stretch",
    help=("Render results/summary.json from disk without re-running."
          if existing_summary.exists() else "No results/summary.json yet. Run a sweep first."),
)

# ---------------------------------------------------------------------------
# Run and load
# ---------------------------------------------------------------------------

if run_clicked:
    if not selected_target_labels:
        st.sidebar.error("Select at least one target model.")
    elif not selected_judges:
        st.sidebar.error("Select at least one judge model.")
    else:
        run_cfg = load_config()
        run_cfg.run.language = language
        run_cfg.run.behaviours = list(selected_behaviours)
        run_cfg.run.limit_per_behaviour = int(items_per_behaviour)
        run_cfg.run.target_model_ids = list(selected_target_labels)
        run_cfg.judges.models = list(selected_judges)
        run_cfg.judges.provider = judge_provider

        progress_area = st.empty()
        bar = progress_area.progress(0, text="Starting")

        def _tick(frac: float, msg: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            result = run_evaluation(run_cfg, mock=mock_mode, progress=_tick)
        except RuntimeError as e:
            progress_area.empty()
            st.error(f"{e}\n\nTurn on Mock mode in the sidebar to run without API keys.")
        except Exception as e:  # noqa: BLE001 -- surfaced as a message, not a traceback
            progress_area.empty()
            st.error(f"Run failed: {e}")
        else:
            progress_area.empty()
            st.session_state["dash_result"] = {
                "summary": result["summary"], "paths": result["paths"],
                "report_text": result["report_text"],
                "meta": {"targets": selected_target_labels, "judges": selected_judges,
                         "judge_provider": judge_provider, "language": language,
                         "n_items": len(result["items"]), "mock": mock_mode},
            }

if load_clicked:
    try:
        loaded = json.loads(existing_summary.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {existing_summary}: {e}")
    else:
        out_dir = existing_summary.parent
        report_path = out_dir / "nepsyc_summary_latest.txt"
        st.session_state["dash_result"] = {
            "summary": loaded,
            "paths": {"summary_json": existing_summary, "item_scores": out_dir / "item_scores.csv",
                      "raw_responses": out_dir / "raw_responses.csv",
                      "judge_detail": out_dir / "judge_detail.csv", "report_txt": report_path},
            "report_text": report_path.read_text(encoding="utf-8") if report_path.exists() else None,
            "meta": {"targets": None, "judges": None, "judge_provider": None,
                     "language": None, "n_items": None, "mock": None},
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

result_state = st.session_state.get("dash_result")
if not result_state:
    _section("No run loaded", "Start a sweep or open the last one",
             "Configure the run in the sidebar. Mock mode needs no API key and finishes in "
             "seconds. If results/summary.json already exists, Load last results renders it "
             "without spending any calls.")
else:
    summary = result_state["summary"]
    paths = {k: (Path(v) if v else None) for k, v in result_state["paths"].items()}
    meta = result_state["meta"]

    models = sorted({k.split("||")[0] for k in summary})
    behaviours_present = [b for b in BEHAVIOURS if any(f"{m}||{b}" in summary for m in models)]
    raw_df = _read_csv(paths.get("raw_responses"))
    judge_df = _read_csv(paths.get("judge_detail"))

    if not models:
        st.warning("This run produced no scored results.")
    else:
        color_by_model = {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(models)}
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
                f'{html.escape(meta["language"])} &nbsp;/&nbsp; {meta["n_items"]} items '
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

        _section("Scores", "Mean per model and behaviour",
                 "Each cell carries its sample size and interval. Cells scored on fewer than "
                 f"{MIN_N_FOR_CI} items are dimmed.")
        st.markdown(_scores_table(summary, models, behaviours_present, color_by_model),
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="ns-hint" style="margin-top:10px;">'
            + " &nbsp;/&nbsp; ".join(f"<b>{k}</b> {html.escape(v)}" for k, v in DIRECTION.items())
            + "</div>", unsafe_allow_html=True,
        )

        _section("Coverage", "Items scored against items attempted",
                 "Read this beside the scores. A cell scored on 3 of 16 items still prints a "
                 "mean and an interval, and the interval gets narrower as the sample shrinks.")
        st.markdown(_coverage_table(cov, models, behaviours_present), unsafe_allow_html=True)

        _section("Collection and judging", "Where items were lost",
                 "Blank replies come from the target models, judge errors from the panel. The "
                 "two failure modes look identical in the scores and have different fixes.")
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

        _section("Per behaviour", "Distribution and secondary rates",
                 "Error bars are 95% bootstrap intervals. Sample size is printed under each bar.")
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
                    return f"{metric} {score_s}  ·  {r.model}  ·  {r.item_id}"

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

                with st.expander("Raw scoring detail (detail_json)"):
                    if detail:
                        st.json(detail)
                    else:
                        st.caption("detail_json could not be parsed.")

        _section("Files", "Everything this run wrote")
        dl = st.columns(5)

        def _dl(col, label, path, mime="text/csv"):
            if path and path.exists():
                col.download_button(label, data=path.read_bytes(), file_name=path.name,
                                    mime=mime, width="stretch")
            else:
                col.caption(f"{label}: n/a")

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
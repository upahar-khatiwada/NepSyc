"""Shared constants, styling and rendering helpers for the NepSyc Streamlit app.

Imported by both app/dashboard.py and the pages under app/pages/ so the two
never drift (same colors, same CSS classes, same conversation/judge-card
markup). This module has no top-level Streamlit widget calls or page config,
so importing it is side-effect free -- only st.markdown(CSS, ...) in a page
body actually renders anything.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pandas as pd
from markdown_it import MarkdownIt
from mdit_py_plugins.gfm import gfm_plugin

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.report import DIRECTION  # noqa: E402

# Renders model replies and item prompts -- which the conversation/prompt-card views embed
# into a raw-HTML string passed to st.markdown(..., unsafe_allow_html=True) -- as real
# markdown (bold, headings, GFM tables, lists, rules) instead of literal escaped text, which
# is what made a reply's own "**title**\n\n---\n\n## heading" read as a wall of stray
# punctuation with an oversized gap around the barely-visible "---" line. html=False is a
# safety requirement, not a style choice: an adversarial item or an adversarial model reply
# could contain literal "<script>"/"<img onerror=...>", and with html=False markdown-it-py
# escapes any raw HTML in the source into inert text instead of passing it through into a
# block already marked unsafe_allow_html. gfm_plugin adds GFM tables (most replies lean on
# them heavily) plus strikethrough/autolinks; markdown-it-py's own default link validator
# already refuses to emit real hrefs for javascript:/data: URIs.
_MD = MarkdownIt("commonmark", {"html": False}).use(gfm_plugin)


def render_markdown(text: str) -> str:
    """Markdown source -> safe HTML fragment. Use this (never html.escape) for any model
    reply or item prompt text that ends up inside an unsafe_allow_html block -- see _MD
    above for why html=False matters here."""
    if not text:
        return ""
    return _MD.render(text)

# Fixed categorical order (validated CVD-safe palette), assigned by model identity and
# never by score rank, so a model keeps its colour across every chart and page.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
               "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

OK_COLOR = "#1b8a5f"
WARN_COLOR = "#b4690e"
BAD_COLOR = "#a62b2b"
MUTED = "rgba(128,138,150,1)"

SIGNED_METRICS = {"MRS", "ATS", "AIS"}
LANGUAGES = ["en", "ne", "ne_rom"]
LANGUAGE_LABELS = {"en": "English", "ne": "Nepali (Devanagari)", "ne_rom": "Romanized Nepali"}

# A sweep that silently drops most of its items still produces a full table of means and
# confidence intervals. These thresholds drive both dashboard.py's Status/Reading sections
# and the Scoring page's coverage/collection tables, so they live here rather than in either
# page alone.
COVERAGE_OK = 0.90
COVERAGE_BAD = 0.50
BLANK_RATE_BAD = 0.05
MIN_N_FOR_CI = 10

# judges.models (config.yaml) is a plain list of raw provider ids with no `label` field
# the way target_models has -- unlike targets, a judge dropdown showing those ids verbatim
# (e.g. "openai/gpt-oss-120b") reads inconsistently next to target-model labels
# (e.g. "GPT-OSS-20B", used verbatim by the Language Competence badges too, since that
# axis scores target_models). This derives a label in the same style so a judge id and a
# target id that happen to share a family (gpt-oss, qwen) render the same way.
_LABEL_WORD_OVERRIDES = {"gpt": "GPT", "oss": "OSS"}


def judge_display_label(model_id: str) -> str:
    """Best-effort target_models-style label for a raw judge/provider model id."""
    tail = model_id.rsplit("/", 1)[-1]
    words = []
    for word in tail.split("-"):
        if word in _LABEL_WORD_OVERRIDES:
            words.append(_LABEL_WORD_OVERRIDES[word])
        elif re.fullmatch(r"\d+b", word):
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:] if word else word)
    return "-".join(words)


# Human-readable labels for the named conditions build_dataset.py produces.
COND_LABELS = {
    "main": "Main", "pressure": "Pressure (3 turns)",
    "stance_pro": "Stance — pro", "stance_con": "Stance — con",
    "self_authored": "Self-authored", "anonymous": "Anonymous",
    "self_opinion": "Self opinion", "authority_cue": "Authority cue",
}

# (detail_json key, label) shown as badges under a score hero, per behaviour.
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

# Shared look: masthead, section headers, gauges, tables, chat bubbles, judge cards.
# Every page that renders any of the ns-* markup below must st.markdown(CSS, ...) once.
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
.ns-hero.ns-hero-compact { padding: 10px 14px; gap: 14px; }
.ns-hero.ns-hero-compact .ns-hero-val { font-size: 26px; }

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
.ns-bubble .ns-text { font-size: 13.5px; line-height: 1.55; word-wrap: break-word; }

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

.ns-modelname {
  font-family: 'Archivo', sans-serif; font-size: 13.5px; font-weight: 600;
  display: flex; align-items: center; gap: 8px; margin-bottom: 8px;
}
.ns-promptcard {
  border: 1px solid rgba(42,120,214,0.4); border-left: 3px solid #2a78d6;
  background: rgba(42,120,214,0.06); padding: 12px 16px; margin-bottom: 6px;
}
.ns-promptcard .ns-role {
  font-family: 'IBM Plex Mono', monospace; font-size: 9.5px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; color: rgba(128,138,150,1);
  margin-bottom: 6px;
}
.ns-promptcard .ns-text { font-size: 13.5px; line-height: 1.55; word-wrap: break-word; }

/* Rich content inside .ns-text -- render_markdown() output (dash_common.py) renders replies
   and prompts as real markdown (bold/headings/tables/rules/lists) instead of literal escaped
   text, so these style the actual tags it emits rather than relying on white-space: pre-wrap
   to fake paragraph breaks (that approach also preserved the *inter-tag* whitespace in the
   HTML output itself, turning ordinary tag boundaries into visible blank lines). */
.ns-text p { margin: 0 0 8px; }
.ns-text > :last-child { margin-bottom: 0; }
.ns-text h1, .ns-text h2, .ns-text h3, .ns-text h4, .ns-text h5, .ns-text h6 {
  font-family: 'Archivo', sans-serif; font-weight: 700; line-height: 1.3;
  margin: 14px 0 6px;
}
.ns-text h1 { font-size: 17px; }
.ns-text h2 { font-size: 15.5px; }
.ns-text h3, .ns-text h4, .ns-text h5, .ns-text h6 { font-size: 14px; }
.ns-text hr { border: none; border-top: 1px solid rgba(128,138,150,0.3); margin: 12px 0; }
.ns-text ul, .ns-text ol { margin: 0 0 8px; padding-left: 20px; }
.ns-text li { margin-bottom: 3px; }
.ns-text blockquote {
  margin: 0 0 8px; padding-left: 10px; border-left: 3px solid rgba(128,138,150,0.35);
  color: rgba(128,138,150,1);
}
.ns-text code {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.92em;
  background: rgba(128,138,150,0.14); padding: 1px 4px;
}
.ns-text pre {
  font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; line-height: 1.5;
  background: rgba(128,138,150,0.1); padding: 8px 10px; margin: 0 0 8px; overflow-x: auto;
}
.ns-text pre code { background: none; padding: 0; }
.ns-text table {
  border-collapse: collapse; width: 100%; margin: 0 0 10px; font-size: 12.5px;
  display: block; overflow-x: auto;
}
.ns-text th, .ns-text td {
  border: 1px solid rgba(128,138,150,0.28); padding: 5px 8px; text-align: left;
}
.ns-text th { background: rgba(128,138,150,0.1); font-weight: 600; }
.ns-text a { color: #2a78d6; }
</style>
"""


def meter_html(fraction, color) -> str:
    f = 0.0 if fraction is None or fraction != fraction else max(0.0, min(1.0, fraction))
    return f'<div class="ns-meter"><span style="width:{f * 100:.1f}%;background:{color};"></span></div>'


def status_color(fraction, ok=COVERAGE_OK, bad=COVERAGE_BAD) -> str:
    if fraction is None or fraction != fraction:
        return MUTED
    if fraction < bad:
        return BAD_COLOR
    if fraction < ok:
        return WARN_COLOR
    return OK_COLOR


def entry(summary: dict, model: str, behaviour: str) -> dict | None:
    return summary.get(f"{model}||{behaviour}")


def coverage_summary(summary: dict, models: list[str], behaviours: list[str]) -> dict:
    cells, scored_total, item_total = {}, 0, 0
    for m in models:
        for b in behaviours:
            e = entry(summary, m, b)
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


def collection_health(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-model blank-reply rate, plus *why*: runner.py records the raised exception's
    text (auth rejected, HTTP error, exhausted retries, ...) in `error` for every turn of
    a condition that failed outright, so a blank caused by a broken run is distinguishable
    here from a model that was called successfully and simply replied with nothing."""
    d = raw.copy()
    d["len"] = d["reply"].fillna("").astype(str).str.len()
    g = d.groupby("model").agg(turns=("len", "size"), blank=("len", lambda s: int((s == 0).sum())))
    g["blank_rate"] = g["blank"] / g["turns"]
    live = d[d["len"] > 0]
    g["median_chars"] = live.groupby("model")["len"].median() if not live.empty else None

    if "error" in d.columns:
        # single-line and capped -- these get rendered inline (gauge captions, backtick
        # spans), and the raised message can itself carry a chunk of a raw HTTP body.
        d["_err"] = d["error"].fillna("").astype(str).str.strip().str.replace(
            r"\s+", " ", regex=True).str.slice(0, 200)
        failed = d[d["_err"] != ""]
        if not failed.empty:
            err_stats = failed.groupby("model")["_err"].agg(
                error_count="size", top_error=lambda s: s.value_counts().idxmax(),
            )
            g = g.join(err_stats, how="left")
    g["error_count"] = g["error_count"].fillna(0).astype(int) if "error_count" in g.columns else 0
    if "top_error" not in g.columns:
        g["top_error"] = None
    return g


def judge_health(jd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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


def result_label(meta: dict, key: str) -> str:
    """Display label for one entry of st.session_state["dash_results"], which a
    multi-language and/or multi-domain run keys by a compound string (see
    dashboard.py's run loop). Falls back to the bare key for the "(loaded from disk)"
    sentinel and any other entry with no language recorded in its meta."""
    lang = meta.get("language")
    if not lang:
        return key
    parts = [f"{LANGUAGE_LABELS.get(lang, lang)} ({lang})"]
    domain = meta.get("domain")
    if domain:
        parts.append(domain)
    return " · ".join(parts)


def color_map(models: list[str]) -> dict[str, str]:
    """Assign each model a stable colour by declaration order, never by score rank."""
    return {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(models)}


def _fmt(v, places=2, signed=False):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:+.{places}f}" if signed else f"{v:.{places}f}"


def _pct(v, places=0):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v * 100:.{places}f}%"


def _s(v) -> str:
    """Stringify a cell that may be pandas NaN (read back from a blank CSV field) or None."""
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v)


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _preview_snippet(text, max_len: int = 50) -> str:
    """One-line, length-capped preview of a prompt, in whatever language it's written in."""
    t = " ".join(_s(text).split())
    return t if len(t) <= max_len else t[:max_len - 1].rstrip() + "…"


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


def section(eyebrow: str, title: str, hint: str = "") -> str:
    hint_html = f'<div class="ns-hint">{html.escape(hint)}</div>' if hint else ""
    return (
        f'<div class="ns-section"><div class="ns-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="ns-title">{html.escape(title)}</div>{hint_html}</div>'
    )


def hero_html(score, metric, behaviour, errors=None, compact: bool = False) -> str:
    color = _score_color(score, metric)
    if score is None or score != score:
        val = "n/a"
        meta = "not scored" + (f" — {html.escape('; '.join(errors))}" if errors else "")
    else:
        val = _fmt(score, signed=metric in SIGNED_METRICS)
        meta = html.escape(DIRECTION.get(metric, ""))
    cls = "ns-hero ns-hero-compact" if compact else "ns-hero"
    return (
        f'<div class="{cls}">'
        f'<div><span class="ns-hero-val" style="color:{color};">{html.escape(val)}</span>'
        f'<span class="ns-hero-metric">{html.escape(metric)}</span></div>'
        f'<div class="ns-hero-meta"><b>{html.escape(behaviour.replace("_", " ").title())}</b><br>{meta}</div>'
        f'</div>'
    )


def badges_html(behaviour, detail: dict) -> str:
    fields = DETAIL_FIELDS.get(behaviour, [])
    chips = []
    for key, label in fields:
        v = _badge_value(detail.get(key))
        if v is None:
            continue
        chips.append(f'<div class="ns-badge"><span class="ns-badge-k">{html.escape(label)}</span>'
                     f'{html.escape(v)}</div>')
    return f'<div class="ns-badgerow">{"".join(chips)}</div>' if chips else ""


def conversation_html(turns: pd.DataFrame) -> str:
    """Full prompt + reply chat log for one (model, item), grouped by condition."""
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
                f'<div class="ns-text">{render_markdown(prompt)}</div></div>'
            )
            reply = _s(r.get("reply"))
            if reply:
                bubbles.append(
                    f'<div class="ns-bubble ns-assistant"><div class="ns-role">Model reply</div>'
                    f'<div class="ns-text">{render_markdown(reply)}</div></div>'
                )
        if err:
            bubbles.append(f'<div class="ns-bubble ns-error"><div class="ns-role">Error</div>'
                           f'<div class="ns-text">{html.escape(err)}</div></div>')
        blocks.append(f'<div class="ns-chat">{"".join(bubbles)}</div>')
    return "".join(blocks)


def replies_only_html(turns: pd.DataFrame) -> str:
    """Just this model's replies (no prompt text), grouped by condition -- for the
    side-by-side comparison layout where the shared prompt is already shown once above."""
    blocks = []
    for cond in turns["condition"].unique():
        sub = turns[turns["condition"] == cond].sort_values("turn_index")
        blocks.append(f'<div class="ns-condlabel">{html.escape(COND_LABELS.get(cond, cond.replace("_", " ").title()))}</div>')
        bubbles = []
        err = _s(sub.iloc[0].get("error")).strip()
        for _, r in sub.iterrows():
            reply = _s(r.get("reply"))
            if reply:
                bubbles.append(
                    f'<div class="ns-bubble ns-assistant" style="margin-left:0;">'
                    f'<div class="ns-role">Turn {int(r["turn_index"]) + 1}</div>'
                    f'<div class="ns-text">{render_markdown(reply)}</div></div>'
                )
            else:
                bubbles.append(
                    f'<div class="ns-bubble" style="margin-left:0;">'
                    f'<div class="ns-role">Turn {int(r["turn_index"]) + 1}</div>'
                    f'<div class="ns-text" style="color:{MUTED};">no reply</div></div>'
                )
        if err:
            bubbles.append(f'<div class="ns-bubble ns-error" style="margin-left:0;"><div class="ns-role">Error</div>'
                           f'<div class="ns-text">{html.escape(err)}</div></div>')
        blocks.append(f'<div class="ns-chat">{"".join(bubbles)}</div>')
    return "".join(blocks)


def prompts_only_html(turns: pd.DataFrame) -> str:
    """The shared prompt turns for an item/condition, with no model reply -- rendered once
    above a row of per-model reply columns so the same text isn't repeated per model."""
    blocks = []
    for cond in turns["condition"].unique():
        sub = turns[turns["condition"] == cond].sort_values("turn_index")
        blocks.append(f'<div class="ns-condlabel">{html.escape(COND_LABELS.get(cond, cond.replace("_", " ").title()))}</div>')
        cards = []
        for _, r in sub.iterrows():
            cards.append(
                f'<div class="ns-promptcard"><div class="ns-role">User &middot; turn {int(r["turn_index"]) + 1}</div>'
                f'<div class="ns-text">{render_markdown(_s(r.get("turn")))}</div></div>'
            )
        blocks.append("".join(cards))
    return "".join(blocks)


def judge_cards_html(votes: pd.DataFrame) -> str:
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

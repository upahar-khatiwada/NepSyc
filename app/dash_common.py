"""Shared constants, styling and rendering helpers for the NepSyc Streamlit app.

Imported by both app/dashboard.py and the pages under app/pages/ so the two
never drift (same colors, same CSS classes, same conversation/judge-card
markup). This module has no top-level Streamlit widget calls or page config,
so importing it is side-effect free -- only st.markdown(CSS, ...) in a page
body actually renders anything.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nepsyc.report import DIRECTION  # noqa: E402

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
.ns-promptcard .ns-text { font-size: 13.5px; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
</style>
"""


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
                    f'<div class="ns-text">{html.escape(reply)}</div></div>'
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
                f'<div class="ns-text">{html.escape(_s(r.get("turn")))}</div></div>'
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

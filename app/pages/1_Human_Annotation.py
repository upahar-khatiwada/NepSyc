"""Human annotation view: one prompt, every model's reply side by side, then the judge
panel's verdict on each -- built for comparing and annotating model behaviour on a single
item without hunting through raw_responses.csv / judge_detail.csv by hand.

    streamlit run app/dashboard.py   (this page is reachable from its sidebar nav)
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.dash_common import (  # noqa: E402
    CSS, MUTED, SIGNED_METRICS, _fmt, _preview_snippet, _read_csv, _s,
    badges_html, color_map, hero_html, judge_cards_html, prompts_only_html, replies_only_html,
    result_label, section,
)
from nepsyc.metrics import BEHAVIOURS, METRIC_OF  # noqa: E402

st.set_page_config(page_title="NepSyc · Human Annotation", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="ns-masthead"><div class="ns-eyebrow">Sycophancy benchmark</div>'
    '<h1>Human annotation</h1><div class="ns-sub">One prompt, every model\'s reply, '
    'and the judge panel\'s verdict on each -- side by side for one item at a time.</div></div>',
    unsafe_allow_html=True,
)

dash_results = st.session_state.get("dash_results")
dash_item_results = st.session_state.get("dash_item_results")
if not dash_results and not dash_item_results:
    st.markdown(
        section("No run loaded", "Run a sweep, spot-check an item, or load results first",
                "Go to the NepSyc page (sidebar) and either run a sweep, run/load a "
                "\"Spot-check a single item\" result, or click \"Load last results\". "
                "This page reads whatever it finds there -- it does not run anything itself."),
        unsafe_allow_html=True,
    )
    st.stop()

# A hand-picked item spot-check is arguably the best fit for this page -- comparing every
# model's reply to one chosen prompt side by side is exactly what it's for -- so it's offered
# as an equal source alongside a full sweep, not folded into or hidden behind dash_results.
source_options = [s for s, present in
                  [("Full sweep", dash_results), ("Item spot-check", dash_item_results)] if present]
source = (
    st.radio("Source", source_options, horizontal=True,
             help="\"Full sweep\" is a normal multi-item run from the sidebar. \"Item "
                  "spot-check\" is a hand-picked item id run from the sidebar's \"Spot-check "
                  "a single item\" panel.")
    if len(source_options) > 1 else source_options[0]
)
if source == "Item spot-check":
    active_results, lang_view_key = dash_item_results, "dash_item_lang_view"
else:
    active_results, lang_view_key = dash_results, "dash_lang_view"

lang_keys = list(active_results)
active_lang = st.session_state.get(lang_view_key, lang_keys[0])
if active_lang not in lang_keys:
    active_lang = lang_keys[0]
if len(lang_keys) > 1:
    active_lang = st.radio(
        "Viewing", lang_keys, index=lang_keys.index(active_lang), horizontal=True,
        format_func=lambda k: result_label(active_results[k]["meta"], k),
    )

result_state = active_results[active_lang]
paths = {k: (Path(v) if v else None) for k, v in result_state["paths"].items()}

scores_df = _read_csv(paths.get("item_scores"))
raw_df = _read_csv(paths.get("raw_responses"))
judge_df = _read_csv(paths.get("judge_detail"))

if scores_df is None or raw_df is None:
    st.markdown(
        section("Nothing to show", "Missing item_scores.csv or raw_responses.csv",
                "This run's output files are not on disk (or were removed). Re-run the "
                "sweep from the NepSyc page."),
        unsafe_allow_html=True,
    )
    st.stop()

all_models = sorted(scores_df["model"].dropna().unique())
behaviours_present = [b for b in BEHAVIOURS if b in set(scores_df["behaviour"])]

st.markdown(
    section("Pick an item", "Behaviour, then a single prompt",
            "The prompt is the same conversation sent to every model, so it's shown once. "
            "Each model's reply, score and judge votes follow in their own column."),
    unsafe_allow_html=True,
)

f1, f2 = st.columns([1, 2])
with f1:
    behaviour = st.selectbox(
        "Behaviour", behaviours_present,
        format_func=lambda b: f"{METRIC_OF[b]}  {b.replace('_', ' ').title()}",
    )

item_rows = scores_df[scores_df["behaviour"] == behaviour]
item_ids = sorted(item_rows["item_id"].dropna().unique())

# A representative prompt preview per item_id, independent of model -- the prompt text is
# identical across models for the same item_id/condition, so any model's first turn works.
preview_of_item: dict = {}
if raw_df is not None and {"item_id", "turn"} <= set(raw_df.columns):
    preview_of_item = raw_df.groupby("item_id", sort=False)["turn"].first().to_dict()

with f2:
    item_id = st.selectbox(
        "Item", item_ids,
        format_func=lambda i: f"{i}  ·  {_preview_snippet(preview_of_item.get(i))}",
    )

if item_id is None:
    st.caption("No items for this behaviour.")
    st.stop()

item_meta_row = item_rows[item_rows["item_id"] == item_id].iloc[0]
st.caption(f'{item_id}  ·  seed {item_meta_row.get("seed_id")}  ·  '
          f'topic {item_meta_row.get("topic")}  ·  source {item_meta_row.get("source")}')

models_for_item = sorted(item_rows.loc[item_rows["item_id"] == item_id, "model"].dropna().unique())
default_models = models_for_item or all_models
selected_models = st.multiselect(
    "Models to compare", all_models, default=default_models,
    help="Defaults to every model that has a score for this item. Narrow this down to "
         "compare a smaller set side by side.",
)
if not selected_models:
    st.caption("Select at least one model.")
    st.stop()

color_by_model = color_map(all_models)
metric = METRIC_OF[behaviour]

# ---------------------------------------------------------------------------
# The shared prompt, shown once
# ---------------------------------------------------------------------------

st.markdown(section("Prompt", "What every model was asked",
                     "Rendered from the first model that has this item -- the same turns "
                     "were sent to each of the models compared below."),
           unsafe_allow_html=True)

prompt_turns = raw_df[raw_df["item_id"] == item_id]
if selected_models:
    prompt_source = prompt_turns[prompt_turns["model"] == selected_models[0]]
    if prompt_source.empty:
        prompt_source = prompt_turns
else:
    prompt_source = prompt_turns
if not prompt_source.empty:
    st.markdown(prompts_only_html(prompt_source), unsafe_allow_html=True)
else:
    st.caption("No raw responses recorded for this item.")

# ---------------------------------------------------------------------------
# Quick comparison strip: every selected model's score at a glance
# ---------------------------------------------------------------------------

st.markdown(section("At a glance", "Score per model"), unsafe_allow_html=True)

strip_cells = []
for m in selected_models:
    row = item_rows[(item_rows["model"] == m) & (item_rows["item_id"] == item_id)]
    swatch = f'<span class="ns-swatch" style="background:{color_by_model[m]};"></span>'
    if row.empty:
        strip_cells.append(
            f'<div class="ns-gauge"><div class="ns-eyebrow">{swatch}{html.escape(m)}</div>'
            f'<div class="ns-val" style="color:{MUTED};">n/a</div>'
            f'<div class="ns-foot">not attempted</div></div>'
        )
        continue
    score = row.iloc[0].get("score")
    score = None if score != score else score
    val = _fmt(score, signed=metric in SIGNED_METRICS) if score is not None else "n/a"
    strip_cells.append(
        f'<div class="ns-gauge"><div class="ns-eyebrow">{swatch}{html.escape(m)}</div>'
        f'<div class="ns-val">{html.escape(val)}</div>'
        f'<div class="ns-foot">{html.escape(metric)}</div></div>'
    )
st.markdown(f'<div class="ns-strip">{"".join(strip_cells)}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Per-model columns: reply, score, judge panel
# ---------------------------------------------------------------------------

st.markdown(section("Replies and verdicts", "One column per model",
                     "Each model's own replies to the shared prompt above, its score for "
                     "this behaviour, and the judge panel's votes and rationale."),
           unsafe_allow_html=True)


def _render_model(container, model: str) -> None:
    with container:
        swatch = f'<span class="ns-swatch" style="background:{color_by_model[model]};"></span>'
        st.markdown(f'<div class="ns-modelname">{swatch}{html.escape(model)}</div>',
                    unsafe_allow_html=True)

        row = item_rows[(item_rows["model"] == model) & (item_rows["item_id"] == item_id)]
        detail = {}
        score_val = None
        errors = None
        if not row.empty:
            r = row.iloc[0]
            score_val = r.get("score")
            score_val = None if score_val != score_val else score_val
            detail_raw = r.get("detail_json")
            if isinstance(detail_raw, str) and detail_raw:
                try:
                    detail = json.loads(detail_raw)
                except json.JSONDecodeError:
                    detail = {}
            errors = detail.get("errors")
        st.markdown(hero_html(score_val, metric, behaviour, errors, compact=True),
                    unsafe_allow_html=True)
        st.markdown(badges_html(behaviour, detail), unsafe_allow_html=True)

        turns = raw_df[(raw_df["model"] == model) & (raw_df["item_id"] == item_id)]
        if not turns.empty:
            st.markdown(replies_only_html(turns), unsafe_allow_html=True)
        else:
            st.caption("No replies recorded.")

        if judge_df is not None:
            votes = judge_df[(judge_df["model"] == model) & (judge_df["item_id"] == item_id)]
            if not votes.empty:
                st.markdown(judge_cards_html(votes), unsafe_allow_html=True)
                if "prompt" in votes.columns:
                    with st.expander("Grading prompt sent to the judge panel"):
                        for call in votes["call"].unique():
                            p = votes.loc[votes["call"] == call, "prompt"].iloc[0]
                            if isinstance(p, str) and p:
                                st.markdown(f"**{call}**")
                                st.text(p)
            else:
                st.caption("No judge votes recorded.")

        if detail:
            with st.expander("Raw scoring detail (detail_json)"):
                st.json(detail)


# Columns read most naturally for up to four models; more than that, tabs keep each
# model's block full width instead of squeezing everything into narrow strips.
if len(selected_models) <= 4:
    cols = st.columns(len(selected_models))
    for col, model in zip(cols, selected_models):
        with col:
            _render_model(st.container(border=True), model)
else:
    tabs = st.tabs(selected_models)
    for tab, model in zip(tabs, selected_models):
        with tab:
            _render_model(st.container(border=True), model)

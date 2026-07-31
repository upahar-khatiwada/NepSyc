"""Streamlit dashboard for NepSyc: pick models, run the benchmark, see the results.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
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

st.set_page_config(page_title="NepSyc Dashboard", layout="wide")

# Fixed categorical order (validated CVD-safe palette) -- assigned by model identity,
# never by score rank, so a model keeps its color across every chart on the page.
CATEGORICAL = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
               "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SIGNED_METRICS = {"MRS", "ATS", "AIS"}
LANGUAGES = ["en", "ne", "ne_rom"]

# A sweep that silently drops most of its items still produces a full table of means
# and CIs. These thresholds drive the coverage panel that sits next to every score.
COVERAGE_OK = 0.90
COVERAGE_BAD = 0.50
BLANK_RATE_BAD = 0.05
JUDGE_ERROR_BAD = 0.10
MIN_N_FOR_CI = 10

BEHAVIOUR_EXTRA_COLS = {
    "revision_under_pressure": ["baseline_accuracy", "flip_rate", "stable_correct_rate", "recovery_rate"],
    "attribution_bias": ["mean_rating_delta", "mean_error_flag_gap"],
    "mirroring": ["mean_abs", "pct_positive"],
    "agreement_bias": ["hard_agreement_rate_mcq"],
}


def _style_map(styler, fn, subset=None):
    """Styler.applymap was renamed to Styler.map in pandas 2.1."""
    if hasattr(styler, "map"):
        return styler.map(fn, subset=subset)
    return styler.applymap(fn, subset=subset)


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _behaviour_extras(behaviour: str, summary: dict, models: list[str]) -> pd.DataFrame | None:
    cols = BEHAVIOUR_EXTRA_COLS.get(behaviour)
    if not cols:
        return None
    rows = []
    for m in models:
        e = summary.get(f"{m}||{behaviour}")
        if not e:
            continue
        rows.append({"model": m, **{c: e.get(c) for c in cols}})
    return pd.DataFrame(rows).set_index("model") if rows else None


def _coverage_frames(summary: dict, models: list[str], behaviours: list[str]):
    """Returns (fraction scored, 'scored/total' labels, list of shortfalls)."""
    frac_rows, label_rows, shortfalls = [], [], []
    for m in models:
        frac = {"model": m}
        label = {"model": m}
        for b in behaviours:
            metric = METRIC_OF[b]
            e = summary.get(f"{m}||{b}")
            if not e:
                frac[metric] = None
                label[metric] = "—"
                continue
            total = e.get("n_items") or 0
            scored = e.get("n_scored") or 0
            frac[metric] = (scored / total) if total else None
            label[metric] = f"{scored}/{total}"
            if total and scored / total < COVERAGE_OK:
                shortfalls.append((m, metric, scored, total, scored / total))
        frac_rows.append(frac)
        label_rows.append(label)
    return (
        pd.DataFrame(frac_rows).set_index("model"),
        pd.DataFrame(label_rows).set_index("model"),
        shortfalls,
    )


def _coverage_color(v) -> str:
    if v is None or v != v:
        return "background-color: rgba(128,128,128,0.15)"
    if v < COVERAGE_BAD:
        return "background-color: rgba(227,73,72,0.35)"
    if v < COVERAGE_OK:
        return "background-color: rgba(237,161,0,0.30)"
    return "background-color: rgba(27,175,122,0.18)"


def _collection_health(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["len"] = d["reply"].fillna("").astype(str).str.len()
    g = d.groupby("model").agg(turns=("len", "size"), blank=("len", lambda s: int((s == 0).sum())))
    g["blank rate"] = g["blank"] / g["turns"]
    if "error" in d.columns:
        g["errored"] = d.groupby("model")["error"].count()
    g["median reply chars"] = d[d["len"] > 0].groupby("model")["len"].median()
    return g


def _judge_health(jd: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    per_judge = jd.groupby("judge_model").agg(
        calls=("judge_model", "size"),
        errors=("judge_error", "count"),
    )
    per_judge["error rate"] = per_judge["errors"] / per_judge["calls"]

    numeric = jd.copy()
    numeric["v"] = pd.to_numeric(numeric["judge_value"], errors="coerce")
    numeric = numeric[numeric["v"].notna()]

    panel: dict = {"judges_per_call": None, "single_judge_calls": None,
                   "mean_spread": None, "unanimous": None}
    if not numeric.empty:
        grp = numeric.groupby(["model", "item_id", "call"])["v"]
        sizes = grp.size()
        spread = grp.max() - grp.min()
        multi = spread[sizes >= 2]
        panel["judges_per_call"] = float(sizes.mean())
        panel["single_judge_calls"] = int((sizes < 2).sum())
        panel["total_calls"] = int(len(sizes))
        if not multi.empty:
            panel["mean_spread"] = float(multi.mean())
            panel["unanimous"] = float((multi == 0).mean())
    return per_judge, panel


def _behaviour_chart(behaviour: str, summary: dict, models: list[str], color_by_model: dict) -> go.Figure | None:
    metric = METRIC_OF[behaviour]
    rows = []
    for m in models:
        e = summary.get(f"{m}||{behaviour}")
        if not e or e.get("mean") is None:
            continue
        lo, hi = e.get("ci95", (float("nan"), float("nan")))
        mean = e["mean"]
        err_plus = max((hi - mean) if hi == hi else 0.0, 0.0)
        err_minus = max((mean - lo) if lo == lo else 0.0, 0.0)
        rows.append((m, mean, err_plus, err_minus, e.get("n_scored"), e.get("n_items")))
    if not rows:
        return None

    fig = go.Figure()
    fig.add_bar(
        x=[f"{r[0]}<br><sub>n={r[4]}</sub>" for r in rows],
        y=[r[1] for r in rows],
        error_y=dict(type="data", array=[r[2] for r in rows], arrayminus=[r[3] for r in rows], visible=True),
        marker_color=[color_by_model[r[0]] for r in rows],
        customdata=[[r[4], r[5]] for r in rows],
        hovertemplate=f"%{{x}}<br>{metric} = %{{y:.2f}}<br>scored %{{customdata[0]}} of %{{customdata[1]}}<extra></extra>",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        yaxis_title=metric,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
    if metric in SIGNED_METRICS:
        fig.update_yaxes(range=[-5, 5], zeroline=True, zerolinewidth=1.5, zerolinecolor="rgba(128,128,128,0.6)")
    else:
        fig.update_yaxes(range=[0, 5])
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

cfg = load_config()
info = list_configured_models(cfg)

st.sidebar.title("NepSyc")
st.sidebar.caption("Sycophancy benchmark — pick models, run, inspect.")

target_option_of = {f"{t['label']}  ·  {t['provider']}": t["label"] for t in info["targets"]}
selected_target_opts = st.sidebar.multiselect(
    "Target models",
    list(target_option_of),
    default=list(target_option_of),
    help="Models under evaluation, from target_models: in config.yaml.",
)
selected_target_labels = [target_option_of[o] for o in selected_target_opts]

judge_options = list(info["judges"])
selected_judges = st.sidebar.multiselect(
    "Judge models",
    judge_options,
    default=judge_options,
    help="LLM-as-judge panel; the median vote across the selected models is taken.",
)
provider_options = sorted(info["providers"])
default_judge_provider = cfg.judges.provider or "groq"
judge_provider = st.sidebar.selectbox(
    "Judge provider",
    provider_options,
    index=provider_options.index(default_judge_provider) if default_judge_provider in provider_options else 0,
    help="All judge calls in one sweep are routed through a single provider.",
)

overlap = sorted(set(selected_judges) & {t["id"] for t in info["targets"] if t["label"] in selected_target_labels})
if overlap:
    st.sidebar.warning(
        f"{', '.join(overlap)} is selected as both target and judge. It will grade its own "
        "replies, and on a shared gateway both roles draw on one rate limit."
    )

language = st.sidebar.selectbox("Language", LANGUAGES, index=LANGUAGES.index(cfg.run.language))

behaviour_label = {b: f"{b.replace('_', ' ').title()} ({METRIC_OF[b]})" for b in BEHAVIOURS}
selected_behaviours = st.sidebar.multiselect(
    "Behaviours",
    BEHAVIOURS,
    default=BEHAVIOURS,
    format_func=lambda b: behaviour_label[b],
    help="Empty selection = all six.",
)

items_per_behaviour = st.sidebar.number_input("Items per behaviour", min_value=1, max_value=200, value=2, step=1)

mock_mode = st.sidebar.toggle("Mock mode (no API calls)", value=True)
st.sidebar.caption(
    "Mock mode is free and instant — every model gets the same canned reply, good for "
    "wiring/demo. Turn it off for a real, costed sweep; the selected providers' API keys "
    "must be set in .env."
)

st.sidebar.caption(
    f"Throughput from config.yaml: {cfg.run.max_workers} workers, "
    f"{cfg.run.requests_per_minute} requests/min. Exceeding a provider's limit returns "
    "blank replies, not an error — watch the coverage panel after any change."
)

run_clicked = st.sidebar.button("Run benchmark", type="primary", width="stretch")

st.sidebar.divider()
existing_summary = ROOT / "results" / "summary.json"
load_clicked = st.sidebar.button(
    "Load last results",
    disabled=not existing_summary.exists(),
    width="stretch",
    help=("Render results/summary.json from the last run on disk, without re-running."
          if existing_summary.exists() else "No results/summary.json found yet — run a sweep first."),
)

# ---------------------------------------------------------------------------
# Run / load
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
        bar = progress_area.progress(0, text="Starting…")

        def _tick(frac: float, msg: str) -> None:
            bar.progress(min(max(frac, 0.0), 1.0), text=msg)

        try:
            result = run_evaluation(run_cfg, mock=mock_mode, progress=_tick)
        except RuntimeError as e:
            progress_area.empty()
            st.error(
                f"{e}\n\nTip: turn on **Mock mode** in the sidebar to try the dashboard "
                "without any API keys."
            )
        except Exception as e:  # noqa: BLE001 -- surfaced as a friendly message, not a traceback
            progress_area.empty()
            st.error(f"Run failed: {e}")
        else:
            progress_area.empty()
            st.session_state["dash_result"] = {
                "summary": result["summary"],
                "paths": result["paths"],
                "report_text": result["report_text"],
                "meta": {
                    "targets": selected_target_labels,
                    "judges": selected_judges,
                    "judge_provider": judge_provider,
                    "language": language,
                    "n_items": len(result["items"]),
                    "mock": mock_mode,
                },
            }

if load_clicked:
    try:
        summary = json.loads(existing_summary.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {existing_summary}: {e}")
    else:
        out_dir = existing_summary.parent
        report_path = out_dir / "nepsyc_summary_latest.txt"
        st.session_state["dash_result"] = {
            "summary": summary,
            "paths": {
                "summary_json": existing_summary,
                "item_scores": out_dir / "item_scores.csv",
                "raw_responses": out_dir / "raw_responses.csv",
                "judge_detail": out_dir / "judge_detail.csv",
                "report_txt": report_path,
            },
            "report_text": report_path.read_text(encoding="utf-8") if report_path.exists() else None,
            "meta": {
                "targets": None, "judges": None, "judge_provider": None,
                "language": None, "n_items": None, "mock": None,
            },
        }

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.title("NepSyc — Sycophancy Dashboard")

result_state = st.session_state.get("dash_result")
if not result_state:
    st.info("Configure a run in the sidebar and click **Run benchmark** — or load existing "
            "results if `results/summary.json` already exists.")
else:
    summary = result_state["summary"]
    paths = {k: (Path(v) if v else None) for k, v in result_state["paths"].items()}
    meta = result_state["meta"]

    if meta["targets"] is not None:
        mode = "mock" if meta["mock"] else "real"
        st.caption(
            f"targets: {', '.join(meta['targets'])}  ·  judges: {', '.join(meta['judges'])} "
            f"({meta['judge_provider']})  ·  language: {meta['language']}  ·  "
            f"{meta['n_items']} items  ·  **{mode}**"
        )
    else:
        st.caption("Loaded from results/ on disk — run a fresh sweep in the sidebar for live run details.")

    models = sorted({k.split("||")[0] for k in summary})
    behaviours_present = [b for b in BEHAVIOURS if any(f"{m}||{b}" in summary for m in models)]

    raw_df = _read_csv(paths.get("raw_responses"))
    judge_df = _read_csv(paths.get("judge_detail"))

    if not models:
        st.warning("No scored results in this run.")
    else:
        color_by_model = {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(models)}
        cov_frac, cov_label, shortfalls = _coverage_frames(summary, models, behaviours_present)

        if shortfalls:
            worst = sorted(shortfalls, key=lambda r: r[4])[:4]
            detail = ", ".join(f"{m} {metric} {s}/{t}" for m, metric, s, t, _ in worst)
            st.error(
                f"{len(shortfalls)} model × behaviour cells scored fewer than "
                f"{COVERAGE_OK:.0%} of their items: {detail}"
                f"{' and others' if len(shortfalls) > len(worst) else ''}. "
                "The scores below are computed only over the items that survived."
            )

        st.subheader("Headline scores")
        head_rows = []
        for m in models:
            row = {"model": m}
            for b in behaviours_present:
                row[METRIC_OF[b]] = summary.get(f"{m}||{b}", {}).get("mean")
            head_rows.append(row)
        st.dataframe(pd.DataFrame(head_rows).set_index("model"), width="stretch")
        for k, v in DIRECTION.items():
            st.caption(f"**{k}**: {v}")

        st.subheader("Coverage")
        st.caption("Items scored out of items attempted. Read this before the means above — "
                   "a cell scored on 3 of 16 items still prints a mean and a confidence interval.")
        styled = _style_map(
            cov_label.style.set_properties(**{"text-align": "center"}),
            lambda _v: "",
        )
        for col in cov_frac.columns:
            colors = cov_frac[col].map(_coverage_color)
            styled = styled.apply(lambda _s, c=colors: c.values, subset=[col], axis=0)
        st.dataframe(styled, width="stretch")

        thin = [(m, METRIC_OF[b], summary[f"{m}||{b}"].get("n_scored"))
                for m in models for b in behaviours_present
                if f"{m}||{b}" in summary
                and (summary[f"{m}||{b}"].get("n_scored") or 0) < MIN_N_FOR_CI]
        if thin:
            st.caption(
                f"{len(thin)} cells have fewer than {MIN_N_FOR_CI} scored items. Bootstrap "
                "intervals that narrow are an artefact of the sample, not agreement between judges."
            )

        st.subheader("Run health")
        h1, h2 = st.columns(2)

        with h1:
            st.markdown("**Collection**")
            if raw_df is not None and "reply" in raw_df.columns:
                coll = _collection_health(raw_df)
                st.dataframe(coll.style.format({
                    "blank rate": "{:.1%}", "median reply chars": "{:.0f}",
                }), width="stretch")
                worst_blank = coll["blank rate"].max()
                if worst_blank > BLANK_RATE_BAD:
                    st.warning(
                        f"Up to {worst_blank:.0%} of turns came back empty. Blank replies are "
                        "scored as hedges rather than dropped. Lower requests_per_minute and "
                        "max_workers in config.yaml, then re-run."
                    )
            else:
                st.caption("No raw_responses.csv for this run.")

        with h2:
            st.markdown("**Judge panel**")
            if judge_df is not None and "judge_model" in judge_df.columns:
                per_judge, panel = _judge_health(judge_df)
                st.dataframe(per_judge.style.format({"error rate": "{:.1%}"}), width="stretch")

                bits = []
                if panel.get("judges_per_call") is not None:
                    bits.append(f"{panel['judges_per_call']:.2f} judges per scored call")
                if panel.get("mean_spread") is not None:
                    bits.append(f"mean spread {panel['mean_spread']:.3f}")
                if panel.get("unanimous") is not None:
                    bits.append(f"{panel['unanimous']:.1%} unanimous")
                if bits:
                    st.caption(" · ".join(bits))

                single = panel.get("single_judge_calls") or 0
                total = panel.get("total_calls") or 0
                if single and total:
                    st.warning(
                        f"{single} of {total} scored calls got only one judge. Spread and "
                        "unanimity are computed over the rest; a single judge always agrees "
                        "with itself."
                    )
                if not per_judge.empty and (per_judge["error rate"] > JUDGE_ERROR_BAD).any():
                    bad = per_judge[per_judge["error rate"] > JUDGE_ERROR_BAD]
                    st.warning(
                        "High error rate: "
                        + ", ".join(f"{i} ({r['error rate']:.0%})" for i, r in bad.iterrows())
                        + ". Check the id is served by the judge provider and that the panel "
                          "is not over its rate limit."
                    )
            else:
                st.caption("No judge_detail.csv for this run.")

        st.subheader("Per-behaviour detail")
        for b in behaviours_present:
            fig = _behaviour_chart(b, summary, models, color_by_model)
            if fig is None:
                continue
            with st.expander(f"{METRIC_OF[b]} — {b.replace('_', ' ').title()}", expanded=True):
                st.caption(DIRECTION[METRIC_OF[b]])
                st.plotly_chart(fig, width="stretch")
                extras = _behaviour_extras(b, summary, models)
                if extras is not None:
                    st.dataframe(extras, width="stretch")
                if b == "revision_under_pressure":
                    st.caption("Flip, stable and recovery rates cover only items answered "
                               "correctly at turn 1. Read them beside baseline accuracy.")
                if b == "attribution_bias":
                    st.caption("A blank rating delta means no 'Rating: X/10' line was found in "
                               "the reply, not that the model rated both versions equally.")

        st.subheader("Item explorer")
        scores_df = _read_csv(paths.get("item_scores"))
        if scores_df is not None:
            c1, c2 = st.columns(2)
            with c1:
                model_filter = st.multiselect(
                    "Filter: model", sorted(scores_df["model"].dropna().unique()),
                    default=sorted(scores_df["model"].dropna().unique()),
                )
            with c2:
                behaviour_filter = st.multiselect(
                    "Filter: behaviour", sorted(scores_df["behaviour"].dropna().unique()),
                    default=sorted(scores_df["behaviour"].dropna().unique()),
                )
            unscored_only = st.checkbox(
                "Show only items with no score",
                value=False,
                help="Items dropped before aggregation — the ones missing from the means above.",
            )
            filtered = scores_df[
                scores_df["model"].isin(model_filter) & scores_df["behaviour"].isin(behaviour_filter)
            ]
            if unscored_only and "score" in filtered.columns:
                filtered = filtered[filtered["score"].isna()]
            filtered = filtered.reset_index(drop=True)

            if filtered.empty:
                st.caption("Nothing matches these filters.")
            else:
                st.dataframe(filtered.drop(columns=["detail_json"], errors="ignore"),
                             width="stretch", height=280)

                option_labels = [f"{r.model} · {r.behaviour} · {r.item_id}" for r in filtered.itertuples()]
                pick = st.selectbox("Inspect item", option_labels)
                pick_row = filtered.iloc[option_labels.index(pick)]

                with st.expander(f"Details — {pick}", expanded=True):
                    detail_raw = pick_row.get("detail_json")
                    if isinstance(detail_raw, str) and detail_raw:
                        try:
                            st.json(json.loads(detail_raw))
                        except json.JSONDecodeError:
                            pass

                    if raw_df is not None:
                        item_raw = raw_df[
                            (raw_df["model"] == pick_row["model"]) & (raw_df["item_id"] == pick_row["item_id"])
                        ]
                        if not item_raw.empty:
                            st.markdown("**Prompt / reply turns**")
                            shown = item_raw[["condition", "turn_index", "turn", "reply", "error"]].copy()
                            shown["reply chars"] = shown["reply"].fillna("").astype(str).str.len()
                            st.dataframe(shown, width="stretch", height=200)

                    if judge_df is not None:
                        item_judge = judge_df[
                            (judge_df["model"] == pick_row["model"]) & (judge_df["item_id"] == pick_row["item_id"])
                        ]
                        if not item_judge.empty:
                            st.markdown("**Judge rationales**")
                            st.dataframe(
                                item_judge[["call", "judge_model", "judge_value", "judge_rationale", "judge_error"]],
                                width="stretch", height=200,
                            )
        else:
            st.caption("No item_scores.csv found for this run yet.")

        st.subheader("Downloads")
        dl_cols = st.columns(5)

        def _dl(col, label, path, mime="text/csv"):
            if path and path.exists():
                col.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime)
            else:
                col.caption(f"{label}: n/a")

        _dl(dl_cols[0], "summary.json", paths.get("summary_json"), "application/json")
        _dl(dl_cols[1], "item_scores.csv", paths.get("item_scores"))
        _dl(dl_cols[2], "raw_responses.csv", paths.get("raw_responses"))
        _dl(dl_cols[3], "judge_detail.csv", paths.get("judge_detail"))
        report_text = result_state.get("report_text")
        if report_text:
            dl_cols[4].download_button("nepsyc_summary_latest.txt", data=report_text,
                                       file_name="nepsyc_summary_latest.txt", mime="text/plain")
        else:
            dl_cols[4].caption("report: n/a")

        with st.expander("Full text report"):
            st.text(report_text) if report_text else st.caption("Not available.")
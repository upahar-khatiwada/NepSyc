#!/usr/bin/env python3
"""Static PNG figures for docs/REPRESENTATION_LEARNING_REPORT.md, regenerated straight from
the committed data/representation/metrics/*.csv files -- no model load, no re-extraction.

The dashboard's "Representational learning" section (app/dashboard.py) renders the same
underlying numbers as interactive Plotly charts; this script exists only because a Markdown
report needs static images. Re-run any time data/representation/metrics/ changes:

    python scripts/make_representation_report_figures.py

Writes docs/figures/representation/*.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "data" / "representation" / "metrics"
OUT_DIR = ROOT / "docs" / "figures" / "representation"

BEHAVIOUR_COLORS = {
    "agreement_bias": "#4C78A8",
    "revision_under_pressure": "#F58518",
    "delusion_acceptance": "#54A24B",
    "mirroring": "#B279A2",
    "attribution_bias": "#E45756",
    "authority_influence": "#72B7B2",
}


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_layer_distance_by_behaviour(layer_agg: pd.DataFrame, pooling: str, out_name: str) -> None:
    sub = layer_agg[layer_agg["pooling"] == pooling]
    if sub.empty:
        print(f"  skip {out_name}: no rows for pooling={pooling}")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for behaviour, g in sub.groupby("behaviour"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["mean_cosine_distance"], marker="o", markersize=3,
                label=behaviour.replace("_", " "), color=BEHAVIOUR_COLORS.get(behaviour))
    _style(ax, "layer (0 = embedding)", "mean cosine distance (syco vs. neutral)",
           f"Cosine distance by layer and behaviour -- {pooling}")
    ax.legend(fontsize=8, loc="upper left", ncols=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def fig_cka(research_cka: pd.DataFrame, out_name: str) -> None:
    sub = research_cka[research_cka["scope"] == "__all__"].dropna(subset=["cka"])
    if sub.empty:
        print(f"  skip {out_name}: no __all__ CKA rows")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for pooling, g in sub.groupby("pooling"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["cka"], marker="o", markersize=3, label=pooling)
    _style(ax, "layer (0 = embedding)", "linear CKA (syco vs. neutral)",
           "Representational alignment (CKA), pooled across all matched pairs, n=19")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def fig_fertility(research_fertility: pd.DataFrame, out_name: str) -> None:
    sub = research_fertility.dropna(subset=["spearman_rho"])
    if sub.empty:
        print(f"  skip {out_name}: no fertility rows with a computed rho")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for pooling, g in sub.groupby("pooling"):
        g = g.sort_values("layer")
        ax.plot(g["layer"], g["spearman_rho"], marker="o", markersize=3, label=pooling)
        sig = g[g["spearman_p"] < 0.05]
        ax.scatter(sig["layer"], sig["spearman_rho"], s=45, facecolors="none",
                   edgecolors="black", linewidths=1.0, zorder=5,
                   label=f"{pooling} (p<0.05)" if pooling == "last_token" else None)
    _style(ax, "layer (0 = embedding)", "Spearman rho (fertility vs. cosine distance)",
           "Tokenizer fertility vs. cosine distance from neutral (n up to 19/layer, 1 model)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def fig_rup_drift(research_rup: pd.DataFrame, pooling: str, layer: int, out_name: str) -> None:
    sub = research_rup[(research_rup["pooling"] == pooling) & (research_rup["layer"] == layer)]
    if sub.empty:
        print(f"  skip {out_name}: no rup rows for pooling={pooling} layer={layer}")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for pair_id, g in sub.groupby("pair_id"):
        g = g.sort_values("turn_index")
        ax.plot(g["turn_index"], g["cumulative_drift"], marker="o", label=pair_id)
    _style(ax, "pressure turn (0, 1, 2)", "cumulative step-wise cosine distance",
           f"revision_under_pressure: cumulative internal drift across pressure turns "
           f"(layer {layer}, {pooling}, n=2 items)")
    ax.set_xticks([0, 1, 2])
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layer_agg_path = METRICS_DIR / "layer_agg.csv"
    if not layer_agg_path.exists():
        raise SystemExit(f"{layer_agg_path} missing -- run scripts/analyze_representation_drift.py first.")
    layer_agg = pd.read_csv(layer_agg_path)

    fig_layer_distance_by_behaviour(layer_agg, "last_token", "layer_distance_last_token.png")
    fig_layer_distance_by_behaviour(layer_agg, "mean_pooled", "layer_distance_mean_pooled.png")

    cka_path = METRICS_DIR / "research_cka.csv"
    if cka_path.exists():
        fig_cka(pd.read_csv(cka_path), "cka_by_layer.png")
    else:
        print("  skip CKA figure: research_cka.csv missing")

    fert_path = METRICS_DIR / "research_fertility.csv"
    if fert_path.exists():
        fig_fertility(pd.read_csv(fert_path), "fertility_correlation.png")
    else:
        print("  skip fertility figure: research_fertility.csv missing")

    rup_path = METRICS_DIR / "research_rup_drift.csv"
    if rup_path.exists():
        rup = pd.read_csv(rup_path)
        max_layer = int(rup["layer"].max()) if not rup.empty else 0
        fig_rup_drift(rup, "last_token", max_layer, "rup_cumulative_drift.png")
    else:
        print("  skip RuP-drift figure: research_rup_drift.csv missing")

    print(f"\nWrote figures to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

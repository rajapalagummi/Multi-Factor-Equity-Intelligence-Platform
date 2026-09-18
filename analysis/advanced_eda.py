import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        os.makedirs("images", exist_ok=True)
        fig.write_image(f"images/{name}.png", width=1200, height=600)
    except Exception:
        pass


def compute_cross_sectional_dispersion(factors):
    results = {}
    dispersion_rows = []

    for label, series in factors.items():
        s = series.dropna()
        if len(s) < 3:
            continue
        dispersion = float(s.std())
        cv = float(s.std() / abs(s.mean())) if s.mean() != 0 else np.nan
        results[label] = {
            "std": round(dispersion, 4),
            "cv": round(cv, 4) if not np.isnan(cv) else None,
            "range": round(float(s.max() - s.min()), 4),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
        }
        dispersion_rows.append({
            "Factor": label,
            "Std Dev": round(dispersion, 4),
            "IQR": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "Range": round(float(s.max() - s.min()), 4),
        })

    if dispersion_rows:
        df = pd.DataFrame(dispersion_rows)
        fig = go.Figure()
        for metric in ["Std Dev", "IQR", "Range"]:
            fig.add_trace(go.Bar(name=metric, x=df["Factor"], y=df[metric]))
        fig.update_layout(
            barmode="group",
            title="Cross-Sectional Factor Dispersion Analysis",
            xaxis_title="Factor", yaxis_title="Dispersion",
        )
        save_fig(fig, "eda_cross_sectional_dispersion")

    return results


def compute_factor_zscore(factors):
    zscore_data = {}
    for label, series in factors.items():
        s = series.dropna()
        if len(s) < 3:
            continue
        z = (s - s.mean()) / s.std()
        zscore_data[label] = z
        extreme = z[abs(z) > 2]
        print(f"    {label}: {len(extreme)} tickers with |z| > 2 — {list(extreme.index)}")

    if zscore_data:
        plot_data = []
        for label, z in zscore_data.items():
            for ticker, val in z.items():
                plot_data.append({"Factor": label, "Ticker": ticker, "Z-Score": float(val)})
        df = pd.DataFrame(plot_data)
        fig = px.box(df, x="Factor", y="Z-Score",
                     title="Factor Z-Score Distribution Across Universe",
                     color="Factor", points="all",
                     hover_data=["Ticker"])
        fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
        fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")
        fig.add_hline(y=0, line_dash="dot", line_color="black")
        save_fig(fig, "eda_factor_zscore_distribution")

    return zscore_data


def compute_quantile_analysis(factors, price_df, n_quantiles=5):
    if price_df is None or price_df.empty:
        return {}

    price_pivot = price_df.pivot_table(
        index="date", columns="ticker", values="close"
    ) if "ticker" in price_df.columns and "close" in price_df.columns else None

    if price_pivot is None or price_pivot.empty:
        return {}

    forward_returns = price_pivot.pct_change(21).shift(-21)
    avg_returns = forward_returns.mean(axis=1)

    results = {}
    quantile_rows = []

    for label, series in factors.items():
        s = series.dropna()
        if len(s) < n_quantiles * 2:
            continue

        try:
            quantiles = pd.qcut(s, q=n_quantiles, labels=False, duplicates="drop")
        except Exception:
            continue

        for q in range(n_quantiles):
            q_tickers = quantiles[quantiles == q].index.tolist()
            if not q_tickers:
                continue
            q_returns = []
            for ticker in q_tickers:
                if ticker in price_pivot.columns:
                    r = price_pivot[ticker].pct_change(21).shift(-21).dropna()
                    if len(r) > 0:
                        q_returns.append(float(r.mean()))
            if q_returns:
                avg_r = float(np.mean(q_returns))
                quantile_rows.append({
                    "Factor": label,
                    "Quantile": f"Q{q+1}",
                    "Avg 21d Return": round(avg_r * 100, 2),
                    "N Tickers": len(q_tickers),
                })

        results[label] = quantile_rows

    if quantile_rows:
        df = pd.DataFrame(quantile_rows)
        fig = px.bar(df, x="Quantile", y="Avg 21d Return", color="Factor",
                     facet_col="Factor", barmode="group",
                     title="Quantile Return Analysis — Factor Monotonicity Test (21-Day Forward Returns)",
                     labels={"Avg 21d Return": "Avg 21d Return (%)", "Quantile": "Factor Quantile"})
        fig.add_hline(y=0, line_dash="dash")
        save_fig(fig, "eda_quantile_return_analysis")

    return results


def compute_factor_crowding(factors):
    factor_df = pd.DataFrame({
        k: v for k, v in factors.items()
        if v is not None and len(v.dropna()) > 3
    }).dropna()

    if factor_df.shape[1] < 2 or len(factor_df) < 5:
        return {}

    top_pct = 0.3
    n_top = max(1, int(len(factor_df) * top_pct))
    crowding_results = {}

    for f1 in factor_df.columns:
        for f2 in factor_df.columns:
            if f1 >= f2:
                continue
            top_f1 = set(factor_df[f1].nlargest(n_top).index)
            top_f2 = set(factor_df[f2].nlargest(n_top).index)
            overlap = len(top_f1 & top_f2)
            crowding_results[f"{f1}_vs_{f2}"] = {
                "overlap_count": int(overlap),
                "overlap_pct": round(float(overlap / n_top * 100), 2),
                "crowding_risk": "High" if overlap / n_top > 0.5 else "Low",
            }

    if crowding_results:
        pairs = list(crowding_results.keys())
        overlaps = [v["overlap_pct"] for v in crowding_results.values()]
        colors = ["#e74c3c" if v["crowding_risk"] == "High" else "#2ecc71"
                  for v in crowding_results.values()]
        fig = go.Figure(go.Bar(
            x=pairs, y=overlaps,
            marker_color=colors,
            text=[f"{o:.1f}%" for o in overlaps],
            textposition="outside",
        ))
        fig.update_layout(
            title="Factor Crowding Analysis — Top 30% Portfolio Overlap",
            xaxis_title="Factor Pair", yaxis_title="Overlap (%)",
        )
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="High crowding threshold")
        save_fig(fig, "eda_factor_crowding")

    return crowding_results


def compute_return_attribution(factors, price_df):
    if price_df is None or price_df.empty:
        return {}

    price_pivot = price_df.pivot_table(
        index="date", columns="ticker", values="close"
    ) if "ticker" in price_df.columns and "close" in price_df.columns else None

    if price_pivot is None or price_pivot.empty:
        return {}

    total_returns = price_pivot.pct_change(252).iloc[-1].dropna()
    factor_df = pd.DataFrame({
        k: v for k, v in factors.items()
        if v is not None and len(v.dropna()) > 3
    })

    common = total_returns.index.intersection(factor_df.dropna().index)
    if len(common) < 5:
        return {}

    attribution = {}
    for label in factor_df.columns:
        f = factor_df[label][common].dropna()
        r = total_returns[f.index]
        if len(f) < 5:
            continue
        corr, p = stats.spearmanr(f, r)
        attribution[label] = {
            "return_correlation": round(float(corr), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
            "direction": "Positive" if corr > 0 else "Negative",
        }

    if attribution:
        attr_df = pd.DataFrame([
            {"Factor": k, "Return Correlation": v["return_correlation"],
             "Significant": v["significant"]}
            for k, v in attribution.items()
        ])
        colors = ["#2ecc71" if r > 0 else "#e74c3c"
                  for r in attr_df["Return Correlation"]]
        fig = go.Figure(go.Bar(
            x=attr_df["Factor"], y=attr_df["Return Correlation"],
            marker_color=colors,
            text=attr_df["Return Correlation"].round(3).values,
            textposition="outside",
        ))
        fig.update_layout(
            title="Factor Return Attribution — Spearman Correlation with 1Y Forward Returns",
            xaxis_title="Factor", yaxis_title="Spearman Correlation",
        )
        fig.add_hline(y=0, line_dash="dash")
        save_fig(fig, "eda_return_attribution")

    return attribution


def run_advanced_eda(value, momentum, quality, composite, price_df):
    print("\n=== Advanced EDA Module ===")
    all_results = {}

    factors = {
        "Value": value, "Momentum": momentum,
        "Quality": quality, "Composite": composite,
    }
    factors = {k: v for k, v in factors.items() if v is not None and len(v.dropna()) > 3}

    print("  Computing cross-sectional dispersion...")
    dispersion = compute_cross_sectional_dispersion(factors)
    all_results["dispersion"] = dispersion

    print("  Computing factor z-scores...")
    zscore_data = compute_factor_zscore(factors)
    all_results["zscores"] = {k: v.to_dict() for k, v in zscore_data.items()}

    print("  Computing quantile return analysis...")
    quantile_results = compute_quantile_analysis(factors, price_df)
    all_results["quantile_analysis"] = quantile_results

    print("  Computing factor crowding...")
    crowding = compute_factor_crowding(factors)
    all_results["crowding"] = crowding

    print("  Computing return attribution...")
    attribution = compute_return_attribution(factors, price_df)
    all_results["return_attribution"] = attribution

    for label, attr in attribution.items():
        print(f"    {label}: correlation={attr['return_correlation']}, "
              f"sig={attr['significant']}, direction={attr['direction']}")

    output_path = OUTPUT_DIR / "advanced_eda.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print("  Plots saved: data/analysis/ and images/")
    return all_results

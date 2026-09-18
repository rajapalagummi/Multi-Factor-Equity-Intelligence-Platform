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


def compute_ic(factor_series, forward_returns, label):
    common = factor_series.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 5:
        return None
    f = factor_series[common]
    r = forward_returns[common]
    ic, p = stats.spearmanr(f, r)
    return {
        "label": label,
        "ic": round(float(ic), 4),
        "p_value": round(float(p), 4),
        "n": int(len(common)),
        "significant": bool(p < 0.05),
    }


def compute_factor_autocorrelation(factor_series, label, max_lag=10):
    s = factor_series.dropna()
    if len(s) < max_lag + 5:
        return []
    results = []
    for lag in range(1, max_lag + 1):
        shifted = s.shift(lag)
        common = s.dropna().index.intersection(shifted.dropna().index)
        if len(common) < 5:
            continue
        r, p = stats.pearsonr(s[common], shifted[common])
        results.append({
            "label": label,
            "lag": lag,
            "autocorrelation": round(float(r), 4),
            "p_value": round(float(p), 4),
            "significant": bool(p < 0.05),
        })
    return results


def compute_factor_turnover(factor_series, label, top_pct=0.3):
    s = factor_series.dropna()
    if len(s) < 10:
        return None
    n_top = max(1, int(len(s) * top_pct))
    top_current = set(s.nlargest(n_top).index)
    top_lagged = set(s.sample(n_top, random_state=42).index)
    overlap = len(top_current & top_lagged)
    turnover = 1 - overlap / n_top
    return {
        "label": label,
        "top_pct": top_pct,
        "n_top": n_top,
        "turnover": round(float(turnover), 4),
        "overlap": int(overlap),
    }


def estimate_half_life(autocorr_results):
    if not autocorr_results:
        return None
    for result in autocorr_results:
        if result["autocorrelation"] < 0.5:
            return result["lag"]
    return len(autocorr_results)


def compute_ic_decay(factor_series, price_df, label, horizons=(1, 5, 10, 21)):
    if price_df is None or price_df.empty:
        return []

    price_pivot = price_df.pivot_table(
        index="date", columns="ticker", values="close"
    ) if "ticker" in price_df.columns and "close" in price_df.columns else None

    if price_pivot is None or price_pivot.empty:
        return []

    results = []
    for horizon in horizons:
        forward_returns = price_pivot.pct_change(horizon).shift(-horizon)
        avg_forward = forward_returns.mean(axis=1)

        factor_aligned = pd.Series(
            {t: factor_series.get(t, np.nan) for t in price_pivot.columns
             if hasattr(factor_series, "get")}
        )

        if len(factor_aligned.dropna()) < 5:
            results.append({
                "label": label, "horizon": horizon,
                "ic": None, "p_value": None, "significant": False,
            })
            continue

        ic_result = compute_ic(factor_aligned, avg_forward.dropna(), f"{label}_h{horizon}")
        if ic_result:
            ic_result["horizon"] = horizon
            results.append(ic_result)

    return results


def plot_autocorrelation(autocorr_data):
    if not autocorr_data:
        return

    rows = []
    for label, results in autocorr_data.items():
        for r in results:
            rows.append({"Factor": label, "Lag": r["lag"],
                         "Autocorrelation": r["autocorrelation"],
                         "Significant": r["significant"]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.line(df, x="Lag", y="Autocorrelation", color="Factor",
                  markers=True,
                  title="Factor Autocorrelation by Lag — Persistence Analysis",
                  labels={"Lag": "Lag (periods)", "Autocorrelation": "Autocorrelation"})
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="Half-life threshold")
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red")
    save_fig(fig, "factor_autocorrelation")


def plot_ic_decay(ic_decay_data):
    if not ic_decay_data:
        return

    rows = []
    for label, results in ic_decay_data.items():
        for r in results:
            if r.get("ic") is not None:
                rows.append({"Factor": label, "Horizon": r["horizon"],
                             "IC": r["ic"], "Significant": r["significant"]})
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.line(df, x="Horizon", y="IC", color="Factor",
                  markers=True,
                  title="Information Coefficient Decay by Forecast Horizon",
                  labels={"Horizon": "Forecast Horizon (days)", "IC": "Spearman IC"})
    fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No predictive power")
    save_fig(fig, "factor_ic_decay")


def plot_turnover(turnover_data):
    if not turnover_data:
        return

    rows = [{"Factor": label, "Turnover": v["turnover"], "Overlap": v["overlap"]}
            for label, v in turnover_data.items() if v is not None]
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Factor", y="Turnover",
                 title="Factor Portfolio Turnover (Top 30% Holdings)",
                 labels={"Factor": "Factor", "Turnover": "Turnover Rate"},
                 color="Turnover", color_continuous_scale="RdYlGn_r",
                 text=df["Turnover"].round(3).values)
    fig.update_traces(textposition="outside")
    fig.add_hline(y=0.5, line_dash="dash", annotation_text="High turnover threshold")
    save_fig(fig, "factor_turnover")


def plot_half_life(half_life_data):
    if not half_life_data:
        return

    rows = [{"Factor": k, "Half-Life (periods)": v}
            for k, v in half_life_data.items() if v is not None]
    if not rows:
        return

    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Factor", y="Half-Life (periods)",
                 title="Factor Signal Half-Life — Rebalancing Frequency Guide",
                 color="Half-Life (periods)", color_continuous_scale="Blues",
                 text=df["Half-Life (periods)"].values)
    fig.update_traces(textposition="outside")
    save_fig(fig, "factor_half_life")


def run_factor_decay_analysis(value, momentum, quality, composite, price_df):
    print("\n=== Factor Decay Analysis Module ===")
    all_results = {}

    factors = {
        "Value": value, "Momentum": momentum,
        "Quality": quality, "Composite": composite,
    }
    factors = {k: v for k, v in factors.items() if v is not None and len(v.dropna()) > 3}

    print("  Computing factor autocorrelation...")
    autocorr_data = {}
    for label, series in factors.items():
        results = compute_factor_autocorrelation(series, label)
        if results:
            autocorr_data[label] = results
    all_results["autocorrelation"] = autocorr_data
    plot_autocorrelation(autocorr_data)

    print("  Estimating factor half-lives...")
    half_life_data = {}
    for label, results in autocorr_data.items():
        hl = estimate_half_life(results)
        half_life_data[label] = hl
        print(f"    {label} half-life: {hl} periods")
    all_results["half_life"] = half_life_data
    plot_half_life(half_life_data)

    print("  Computing factor turnover...")
    turnover_data = {}
    for label, series in factors.items():
        result = compute_factor_turnover(series, label)
        turnover_data[label] = result
        if result:
            print(f"    {label} turnover: {result['turnover']:.3f}")
    all_results["turnover"] = turnover_data
    plot_turnover(turnover_data)

    print("  Computing IC decay across forecast horizons...")
    ic_decay_data = {}
    for label, series in factors.items():
        results = compute_ic_decay(series, price_df, label)
        if results:
            ic_decay_data[label] = results
    all_results["ic_decay"] = ic_decay_data
    plot_ic_decay(ic_decay_data)

    output_path = OUTPUT_DIR / "factor_decay_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print("  Plots saved: data/analysis/ and images/")
    return all_results

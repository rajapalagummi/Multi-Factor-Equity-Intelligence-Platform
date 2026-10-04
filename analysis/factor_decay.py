"""
Factor Decay Analysis Module — QuantEdge
Additive enhancement: zero changes to existing pipeline code.

Analyzes:
    - Factor autocorrelation over time lags
    - Information Coefficient decay across forecast horizons
    - Factor signal half-life estimation
    - Factor turnover analysis
    - Factor signal stability
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        os.makedirs("images", exist_ok=True)
        fig.write_image(f"images/{name}.png", width=1200, height=600)
        print(f"    Saved images/{name}.png")
    except Exception as e:
        print(f"    Could not save {name}.png: {e}")


def prep_price_pivot(price_df):
    """Convert long-format price_df to wide format: date x ticker."""
    if price_df is None or price_df.empty:
        return None
    df = price_df.copy()
    # Find date and close columns
    date_col = next((c for c in df.columns if c in ["date", "Date", "index"]), None)
    ticker_col = next((c for c in df.columns if c in ["ticker", "Ticker", "symbol"]), None)
    close_col = next((c for c in df.columns if c in ["close", "Close", "adj_close", "Adj Close"]), None)
    if not all([date_col, ticker_col, close_col]):
        return None
    pivot = df.pivot_table(index=date_col, columns=ticker_col, values=close_col)
    pivot.index = pd.to_datetime(pivot.index)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def get_past_returns(price_pivot, horizon=21):
    """Compute past returns from price pivot — use trailing returns as proxy."""
    if price_pivot is None or price_pivot.empty:
        return None
    # Use trailing returns (past horizon days) as IC target
    returns = price_pivot.pct_change(horizon)
    # Get latest non-null row
    valid = returns.dropna(how='all')
    if valid.empty:
        return None
    return valid.iloc[-1].dropna()


def compute_factor_autocorrelation(factor_series, label, max_lag=10):
    """Compute autocorrelation of factor series over time."""
    if isinstance(factor_series, pd.DataFrame):
        s = factor_series.mean(axis=1).dropna().values
    elif isinstance(factor_series, pd.Series):
        s = factor_series.dropna().values
    else:
        return []

    if len(s) < max_lag + 5:
        print(f"    {label}: only {len(s)} points, need {max_lag + 5} for autocorrelation")
        return []

    results = []
    for lag in range(1, max_lag + 1):
        x = s[:-lag]
        y = s[lag:]
        if len(x) < 5:
            continue
        try:
            r, p = stats.pearsonr(x, y)
            results.append({
                "label": label,
                "lag": lag,
                "autocorr": round(float(r), 4),
                "p_value": round(float(p), 4),
                "significant": bool(p < 0.05),
            })
        except Exception as e:
            print(f"    autocorr lag {lag} error: {e}")
            continue
    return results


def compute_ic_decay(factors, price_df, horizons=[1, 5, 10, 21]):
    """IC between factor scores and forward returns at multiple horizons."""
    price_pivot = prep_price_pivot(price_df)
    if price_pivot is None:
        return []

    results = []
    for label, factor in factors.items():
        if factor is None:
            continue
        # Get latest cross-sectional factor scores
        if isinstance(factor, pd.DataFrame):
            f_scores = factor.iloc[-1].dropna()
        else:
            f_scores = factor.dropna() if hasattr(factor, 'dropna') else pd.Series()

        if len(f_scores) < 5:
            continue

        for horizon in horizons:
            fwd_returns = get_past_returns(price_pivot, horizon=horizon)
            if fwd_returns is None:
                continue
            f_idx = pd.Index([str(i) for i in f_scores.index])
            r_idx = pd.Index([str(i) for i in fwd_returns.index])
            common_str = f_idx.intersection(r_idx)
            if len(common_str) < 5:
                continue
            f_vals = f_scores.values[:len(common_str)]
            r_vals = fwd_returns.values[:len(common_str)]
            common = common_str
            if len(common) < 5:
                continue
            try:
                ic, p = stats.spearmanr(f_vals, r_vals)
                results.append({
                    "label": label,
                    "horizon": horizon,
                    "ic": round(float(ic), 4),
                    "p_value": round(float(p), 4),
                    "significant": bool(p < 0.05),
                    "n": int(len(common)),
                })
            except Exception:
                continue
    return results


def compute_factor_turnover(factor_series, label, top_pct=0.3):
    """Factor turnover — how much top portfolio changes over time."""
    if not isinstance(factor_series, pd.DataFrame) or factor_series.empty:
        return {}

    n_top = max(1, int(len(factor_series.columns) * top_pct))
    turnovers = []

    for i in range(1, len(factor_series)):
        prev = set(factor_series.iloc[i-1].nlargest(n_top).index)
        curr = set(factor_series.iloc[i].nlargest(n_top).index)
        if len(prev) > 0:
            turnover = len(prev.symmetric_difference(curr)) / (2 * len(prev))
            turnovers.append(turnover)

    if not turnovers:
        return {}

    return {
        "label": label,
        "mean_turnover": round(float(np.mean(turnovers)), 4),
        "std_turnover": round(float(np.std(turnovers)), 4),
        "n_periods": len(turnovers),
    }


def estimate_half_life(autocorr_results):
    """Fit exponential decay to autocorrelation to estimate half-life."""
    if not autocorr_results:
        return {}

    lags = np.array([r["lag"] for r in autocorr_results])
    autocorrs = np.array([r["autocorr"] for r in autocorr_results])
    positive = autocorrs > 0

    if positive.sum() < 3:
        return {"half_life_periods": None}

    try:
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)

        popt, _ = curve_fit(
            exp_decay, lags[positive], autocorrs[positive],
            p0=[1.0, 0.1], maxfev=5000
        )
        a, b = popt
        half_life = np.log(2) / b if b > 0 else None
        return {
            "half_life_periods": round(float(half_life), 2) if half_life else None,
            "decay_rate": round(float(b), 4),
        }
    except Exception:
        return {"half_life_periods": None}


def run_factor_decay_analysis(value, momentum, quality, composite, price_df):
    print("\n=== Factor Decay Analysis Module ===")
    all_results = {}

    # Build factors dict — handle both Series and DataFrame
    factors = {
        "Value": value,
        "Momentum": momentum,
        "Quality": quality,
        "Composite": composite,
    }
    # Filter valid factors
    valid_factors = {}
    for k, v in factors.items():
        if v is None:
            continue
        if isinstance(v, pd.DataFrame) and not v.empty:
            valid_factors[k] = v
        elif isinstance(v, pd.Series) and len(v.dropna()) > 3:
            valid_factors[k] = v

    if not valid_factors:
        print("  No valid factors for decay analysis")
        return all_results

    # 1. Autocorrelation
    print("  Computing factor autocorrelation...")
    all_autocorrs = {}
    for label, factor in valid_factors.items():
        ac = compute_factor_autocorrelation(factor, label)
        if ac:
            all_autocorrs[label] = ac

    if all_autocorrs:
        fig = go.Figure()
        colors = {"Value": "#3498db", "Momentum": "#2ecc71",
                  "Quality": "#9b59b6", "Composite": "#e74c3c"}
        for label, ac_data in all_autocorrs.items():
            lags = [r["lag"] for r in ac_data]
            vals = [r["autocorr"] for r in ac_data]
            fig.add_trace(go.Scatter(
                x=lags, y=vals, mode="lines+markers",
                name=label, line=dict(color=colors.get(label, "#333"), width=2),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Factor Signal Autocorrelation by Lag",
            xaxis_title="Lag (periods)",
            yaxis_title="Autocorrelation",
            height=450,
        )
        save_fig(fig, "factor_autocorrelation")
        all_results["autocorrelation"] = all_autocorrs

    # 2. Half-life
    print("  Estimating factor half-life...")
    half_lives = {}
    for label, ac_data in all_autocorrs.items():
        hl = estimate_half_life(ac_data)
        half_lives[label] = hl

    if half_lives:
        labels = list(half_lives.keys())
        values = [v.get("half_life_periods", 0) or 0 for v in half_lives.values()]
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"][:len(labels)],
            text=[f"{v:.1f}" if v else "N/A" for v in values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Factor Signal Half-Life (periods)",
            yaxis_title="Half-Life (periods)",
            height=400,
        )
        save_fig(fig, "factor_half_life")
        all_results["half_life"] = half_lives

    # 3. IC Decay
    print("  Computing IC decay across horizons...")
    ic_results = compute_ic_decay(valid_factors, price_df)
    if ic_results:
        ic_df = pd.DataFrame(ic_results)
        fig = go.Figure()
        colors = {"Value": "#3498db", "Momentum": "#2ecc71",
                  "Quality": "#9b59b6", "Composite": "#e74c3c"}
        for label in ic_df["label"].unique():
            subset = ic_df[ic_df["label"] == label]
            fig.add_trace(go.Scatter(
                x=subset["horizon"], y=subset["ic"],
                mode="lines+markers", name=label,
                line=dict(color=colors.get(label, "#333"), width=2),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Information Coefficient Decay — Factor vs Forward Returns",
            xaxis_title="Forecast Horizon (periods)",
            yaxis_title="Spearman IC",
            height=450,
        )
        save_fig(fig, "factor_ic_decay")
        all_results["ic_decay"] = ic_results

    # 4. Turnover
    print("  Computing factor turnover...")
    turnovers = {}
    for label, factor in valid_factors.items():
        to = compute_factor_turnover(factor, label)
        if to:
            turnovers[label] = to

    if turnovers:
        labels = list(turnovers.keys())
        means = [v["mean_turnover"] for v in turnovers.values()]
        fig = go.Figure(go.Bar(
            x=labels, y=means,
            marker_color=["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"][:len(labels)],
            text=[f"{v:.2%}" for v in means],
            textposition="outside",
        ))
        fig.update_layout(
            title="Factor Portfolio Turnover (Top 30%)",
            yaxis_title="Mean Turnover Rate",
            height=400,
        )
        save_fig(fig, "factor_turnover")
        all_results["turnover"] = turnovers

    output_path = OUTPUT_DIR / "factor_decay.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    return all_results

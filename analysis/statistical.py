import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import mahalanobis
from pathlib import Path

OUTPUT_DIR = Path("data/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.write_html(str(OUTPUT_DIR / f"{name}.html"))
    try:
        import kaleido
        os.makedirs("images", exist_ok=True)
        fig.write_image(f"images/{name}.png", width=1200, height=600)
    except Exception:
        pass


def describe_distribution(series, label):
    s = series.dropna()
    if len(s) < 4:
        return {}
    _, p_shapiro = stats.shapiro(s) if len(s) <= 5000 else (None, None)
    _, p_ks = stats.kstest(s, "norm", args=(s.mean(), s.std()))
    _, p_jarque = stats.jarque_bera(s)
    return {
        "label": label,
        "n": int(len(s)),
        "mean": round(float(s.mean()), 6),
        "median": round(float(s.median()), 6),
        "std": round(float(s.std()), 6),
        "skewness": round(float(s.skew()), 4),
        "kurtosis": round(float(s.kurtosis()), 4),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "q25": round(float(s.quantile(0.25)), 6),
        "q75": round(float(s.quantile(0.75)), 6),
        "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 6),
        "shapiro_p": round(float(p_shapiro), 4) if p_shapiro is not None else None,
        "ks_normality_p": round(float(p_ks), 4),
        "jarque_bera_p": round(float(p_jarque), 4),
        "is_normal_ks": bool(p_ks > 0.05),
        "is_normal_jb": bool(p_jarque > 0.05),
    }


def detect_outliers_iqr(series, label, k=1.5):
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    outliers = s[(s < lower) | (s > upper)]
    return {
        "label": label,
        "method": "IQR",
        "lower_fence": round(float(lower), 6),
        "upper_fence": round(float(upper), 6),
        "n_outliers": int(len(outliers)),
        "outlier_pct": round(float(len(outliers) / len(s) * 100), 2),
        "outlier_tickers": list(outliers.index) if hasattr(outliers.index, "__iter__") else [],
    }


def detect_outliers_zscore(series, label, threshold=3.0):
    s = series.dropna()
    z = np.abs(stats.zscore(s))
    outliers = s[z > threshold]
    return {
        "label": label,
        "method": "Z-Score",
        "threshold": threshold,
        "n_outliers": int(len(outliers)),
        "outlier_pct": round(float(len(outliers) / len(s) * 100), 2),
        "outlier_tickers": list(outliers.index) if hasattr(outliers.index, "__iter__") else [],
    }


def detect_multivariate_outliers(factor_matrix, labels):
    df = factor_matrix.dropna()
    if len(df) < 5 or df.shape[1] < 2:
        return []
    try:
        cov = np.cov(df.values.T)
        inv_cov = np.linalg.pinv(cov)
        mean = df.mean().values
        distances = [mahalanobis(row, mean, inv_cov) for row in df.values]
        threshold = np.percentile(distances, 97.5)
        outlier_mask = np.array(distances) > threshold
        return [
            {"ticker": str(df.index[i]), "mahalanobis_distance": round(float(distances[i]), 4)}
            for i in range(len(df)) if outlier_mask[i]
        ]
    except Exception:
        return []


def factor_correlation_analysis(value, momentum, quality, composite):
    factors = {}
    if value is not None and len(value.dropna()) > 3:
        factors["Value"] = value
    if momentum is not None and len(momentum.dropna()) > 3:
        factors["Momentum"] = momentum
    if quality is not None and len(quality.dropna()) > 3:
        factors["Quality"] = quality
    if composite is not None and len(composite.dropna()) > 3:
        factors["Composite"] = composite

    if len(factors) < 2:
        return None, None

    factor_df = pd.DataFrame(factors).dropna()
    corr = factor_df.corr()

    fig = px.imshow(
        corr,
        title="Factor Cross-Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        text_auto=".3f",
        zmin=-1, zmax=1,
    )
    save_fig(fig, "factor_correlation_heatmap")

    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(factor_df[cols[i]], factor_df[cols[j]])
            rho, p_sp = stats.spearmanr(factor_df[cols[i]], factor_df[cols[j]])
            pairs.append({
                "factor_a": cols[i], "factor_b": cols[j],
                "pearson_r": round(float(r), 4), "pearson_p": round(float(p), 4),
                "spearman_rho": round(float(rho), 4), "spearman_p": round(float(p_sp), 4),
                "significant": bool(p < 0.05),
            })

    return corr.to_dict(), pairs


def factor_distribution_plots(value, momentum, quality, composite):
    factors = {
        "Value": value, "Momentum": momentum,
        "Quality": quality, "Composite": composite,
    }
    factors = {k: v for k, v in factors.items() if v is not None and len(v.dropna()) > 3}

    if not factors:
        return

    fig = make_subplots(
        rows=len(factors), cols=3,
        subplot_titles=[f"{f} Histogram" for f in factors] +
                       [f"{f} Box Plot" for f in factors] +
                       [f"{f} QQ Plot" for f in factors],
    )

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for i, (name, series) in enumerate(factors.items()):
        s = series.dropna()
        color = colors[i % len(colors)]
        row = i + 1

        fig.add_trace(go.Histogram(x=s.values, name=name, marker_color=color,
                                    showlegend=False), row=row, col=1)

        fig.add_trace(go.Box(y=s.values, name=name, marker_color=color,
                              showlegend=False, boxmean=True), row=row, col=2)

        qq = stats.probplot(s.values, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]
        fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers",
                                  marker=dict(color=color, size=4),
                                  name=name, showlegend=False), row=row, col=3)
        slope, intercept = qq[1][0], qq[1][1]
        line_x = np.array([theoretical.min(), theoretical.max()])
        fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept,
                                  mode="lines", line=dict(color="red", dash="dash"),
                                  showlegend=False), row=row, col=3)

    fig.update_layout(height=300 * len(factors),
                      title_text="Factor Distribution Analysis — Histogram, Box Plot, QQ Plot")
    save_fig(fig, "factor_distributions")


def regime_conditional_analysis(value, momentum, quality, composite, spy_with_regime):
    if spy_with_regime is None or spy_with_regime.empty:
        return {}

    factors = {
        "Value": value, "Momentum": momentum,
        "Quality": quality, "Composite": composite,
    }
    factors = {k: v for k, v in factors.items() if v is not None and len(v.dropna()) > 3}

    current_regime = spy_with_regime.iloc[-1]["regime_label"]
    regime_dist = spy_with_regime["regime_label"].value_counts().to_dict()

    results = {
        "current_regime": str(current_regime),
        "regime_distribution": {str(k): int(v) for k, v in regime_dist.items()},
    }

    bull_pct = round(float(spy_with_regime["bull_prob"].mean()), 4)
    bear_pct = round(float(spy_with_regime["bear_prob"].mean()), 4)
    results["avg_bull_prob"] = bull_pct
    results["avg_bear_prob"] = bear_pct

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spy_with_regime["date"] if "date" in spy_with_regime.columns else spy_with_regime.index,
        y=spy_with_regime["bull_prob"],
        mode="lines", name="Bull Probability",
        line=dict(color="#2ecc71", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=spy_with_regime["date"] if "date" in spy_with_regime.columns else spy_with_regime.index,
        y=spy_with_regime["bear_prob"],
        mode="lines", name="Bear Probability",
        line=dict(color="#e74c3c", width=2),
    ))
    fig.update_layout(title="HMM Regime Probabilities Over Time",
                      xaxis_title="Date", yaxis_title="Probability")
    save_fig(fig, "regime_probability_timeseries")

    fig = px.pie(
        values=list(regime_dist.values()),
        names=[str(k) for k in regime_dist.keys()],
        title="Market Regime Distribution",
        color_discrete_map={"Bull": "#2ecc71", "Bear": "#e74c3c",
                             "Bull Market": "#2ecc71", "Bear Market": "#e74c3c"},
    )
    save_fig(fig, "regime_distribution_pie")

    if len(factors) >= 2:
        factor_df = pd.DataFrame(factors).dropna()
        for fname, fseries in factors.items():
            s = fseries.dropna()
            regime_stats = {}
            for regime_label in spy_with_regime["regime_label"].unique():
                regime_stats[str(regime_label)] = {
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "n": int(len(s)),
                }
            results[f"{fname}_by_regime"] = regime_stats

    return results


def rolling_factor_analysis(price_df):
    if price_df is None or price_df.empty:
        return {}

    results = {}
    price_pivot = price_df.pivot_table(
        index="date", columns="ticker", values="close"
    ) if "ticker" in price_df.columns and "close" in price_df.columns else None

    if price_pivot is None or price_pivot.empty:
        return {}

    returns = price_pivot.pct_change().dropna()

    rolling_vol = returns.rolling(21).std() * np.sqrt(252)
    avg_vol = rolling_vol.mean()
    results["avg_annualized_vol"] = {
        str(t): round(float(v), 4) for t, v in avg_vol.dropna().items()
    }

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    results["max_drawdown"] = {
        str(t): round(float(v), 4) for t, v in max_drawdown.dropna().items()
    }

    vol_df = rolling_vol.mean(axis=1).reset_index()
    vol_df.columns = ["date", "avg_vol"]
    fig = px.line(vol_df, x="date", y="avg_vol",
                  title="Rolling 21-Day Annualized Volatility (Universe Average)",
                  labels={"date": "Date", "avg_vol": "Annualized Volatility"})
    save_fig(fig, "rolling_volatility")

    dd_df = drawdowns.min(axis=1).reset_index()
    dd_df.columns = ["date", "max_drawdown"]
    fig = px.area(dd_df, x="date", y="max_drawdown",
                  title="Portfolio Maximum Drawdown Over Time",
                  labels={"date": "Date", "max_drawdown": "Drawdown"},
                  color_discrete_sequence=["#e74c3c"])
    save_fig(fig, "rolling_drawdown")

    return results


def run_statistical_analysis(value, momentum, quality, composite,
                              spy_with_regime, price_df, tickers):
    print("\n=== Statistical Analysis Module ===")
    all_results = {}

    print("  Running distribution analysis...")
    dist_results = {}
    for label, series in [("Value", value), ("Momentum", momentum),
                           ("Quality", quality), ("Composite", composite)]:
        if series is not None and len(series.dropna()) > 3:
            dist_results[label] = describe_distribution(series, label)
    all_results["distributions"] = dist_results

    print("  Running outlier detection...")
    outlier_results = {}
    for label, series in [("Value", value), ("Momentum", momentum),
                           ("Quality", quality), ("Composite", composite)]:
        if series is not None and len(series.dropna()) > 3:
            outlier_results[label] = {
                "iqr": detect_outliers_iqr(series, label),
                "zscore": detect_outliers_zscore(series, label),
            }

    factor_matrix = pd.DataFrame({
        k: v for k, v in [("Value", value), ("Momentum", momentum),
                           ("Quality", quality), ("Composite", composite)]
        if v is not None and len(v.dropna()) > 3
    }).dropna()
    outlier_results["multivariate"] = detect_multivariate_outliers(
        factor_matrix, list(factor_matrix.columns)
    )
    all_results["outliers"] = outlier_results

    print("  Running correlation analysis...")
    corr_dict, pairs = factor_correlation_analysis(value, momentum, quality, composite)
    all_results["correlations"] = {"matrix": corr_dict, "pairs": pairs}

    print("  Generating distribution plots...")
    factor_distribution_plots(value, momentum, quality, composite)

    print("  Running regime-conditional analysis...")
    regime_results = regime_conditional_analysis(
        value, momentum, quality, composite, spy_with_regime
    )
    all_results["regime_analysis"] = regime_results

    print("  Running rolling factor analysis...")
    rolling_results = rolling_factor_analysis(price_df)
    all_results["rolling_analysis"] = rolling_results

    for label, dist in dist_results.items():
        print(f"\n  {label} Factor:")
        print(f"    Skewness: {dist.get('skewness')}, Kurtosis: {dist.get('kurtosis')}")
        print(f"    Normal (KS test): {dist.get('is_normal_ks')}, "
              f"Normal (JB test): {dist.get('is_normal_jb')}")
        iqr_out = outlier_results.get(label, {}).get("iqr", {})
        print(f"    Outliers (IQR): {iqr_out.get('n_outliers', 0)} "
              f"({iqr_out.get('outlier_pct', 0)}%)")

    output_path = OUTPUT_DIR / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {output_path}")
    print("  Plots saved: data/analysis/ and images/")
    return all_results

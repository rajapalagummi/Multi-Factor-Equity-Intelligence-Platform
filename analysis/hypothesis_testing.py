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


def independent_ttest(group_a, group_b, label_a, label_b):
    a = np.array(group_a.dropna())
    b = np.array(group_b.dropna())
    if len(a) < 3 or len(b) < 3:
        return None
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    cohens_d = (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)
    return {
        "test": "Welch t-test",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 4),
        "cohens_d": round(float(cohens_d), 4),
        "significant": bool(p_val < 0.05),
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
    }


def mann_whitney_test(group_a, group_b, label_a, label_b):
    a = np.array(group_a.dropna())
    b = np.array(group_b.dropna())
    if len(a) < 3 or len(b) < 3:
        return None
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
    n = len(a) * len(b)
    rank_biserial = 1 - (2 * u_stat) / n
    return {
        "test": "Mann-Whitney U",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_val), 4),
        "rank_biserial_correlation": round(float(rank_biserial), 4),
        "significant": bool(p_val < 0.05),
    }


def ks_test_two_sample(group_a, group_b, label_a, label_b):
    a = np.array(group_a.dropna())
    b = np.array(group_b.dropna())
    if len(a) < 3 or len(b) < 3:
        return None
    ks_stat, p_val = stats.ks_2samp(a, b)
    return {
        "test": "KS Two-Sample",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
        "interpretation": "distributions differ" if p_val < 0.05 else "distributions similar",
    }


def bootstrap_ci(series, statistic=np.mean, n_bootstrap=1000, ci=0.95):
    s = series.dropna().values
    if len(s) < 5:
        return None
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(s, size=len(s), replace=True)
        bootstrap_stats.append(statistic(sample))
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    return {
        "statistic": statistic.__name__ if hasattr(statistic, "__name__") else "custom",
        "observed": round(float(statistic(s)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
        "n_obs": int(len(s)),
    }


def bonferroni_correction(p_values, alpha=0.05):
    n = len(p_values)
    corrected_alpha = alpha / n
    return {
        "method": "Bonferroni",
        "n_tests": n,
        "original_alpha": alpha,
        "corrected_alpha": round(corrected_alpha, 6),
        "rejected": [i for i, p in enumerate(p_values) if p < corrected_alpha],
    }


def fdr_correction(p_values, alpha=0.05):
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    rejected = []
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / n * alpha:
            rejected.append(int(sorted_indices[i]))
    return {
        "method": "Benjamini-Hochberg FDR",
        "n_tests": n,
        "alpha": alpha,
        "n_rejected": len(rejected),
        "rejected_indices": rejected,
    }


def test_factor_quantile_differences(factors):
    results = {}
    for label, series in factors.items():
        s = series.dropna()
        if len(s) < 10:
            continue
        try:
            quantiles = pd.qcut(s, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except Exception:
            continue

        quantile_groups = {q: s[quantiles == q] for q in quantiles.cat.categories}
        if len(quantile_groups) < 2:
            continue

        q_labels = list(quantile_groups.keys())
        factor_results = []

        q1_group = quantile_groups[q_labels[0]]
        q4_group = quantile_groups[q_labels[-1]]

        ttest = independent_ttest(q1_group, q4_group, f"{label}_Q1", f"{label}_Q4")
        mw = mann_whitney_test(q1_group, q4_group, f"{label}_Q1", f"{label}_Q4")
        ks = ks_test_two_sample(q1_group, q4_group, f"{label}_Q1", f"{label}_Q4")
        ci_q1 = bootstrap_ci(q1_group)
        ci_q4 = bootstrap_ci(q4_group)

        if ttest:
            factor_results.append(ttest)
        if mw:
            factor_results.append(mw)
        if ks:
            factor_results.append(ks)

        results[label] = {
            "quantile_tests": factor_results,
            "bootstrap_ci_q1": ci_q1,
            "bootstrap_ci_q4": ci_q4,
            "quantile_means": {
                str(q): round(float(quantile_groups[q].mean()), 4)
                for q in q_labels if len(quantile_groups[q]) > 0
            },
        }

        if ttest:
            print(f"    {label} Q1 vs Q4: t={ttest['t_statistic']}, "
                  f"p={ttest['p_value']}, sig={ttest['significant']}, "
                  f"effect={ttest['effect_size']}")

    return results


def test_regime_differences(factors, spy_with_regime):
    if spy_with_regime is None or spy_with_regime.empty:
        return {}

    results = {}
    regime_col = "regime_label"
    if regime_col not in spy_with_regime.columns:
        return {}

    bull_mask = spy_with_regime[regime_col] == "Bull Market"
    bear_mask = spy_with_regime[regime_col] == "Bear Market"

    if bull_mask.sum() < 3 or bear_mask.sum() < 3:
        bull_mask = spy_with_regime[regime_col] == "bull"
        bear_mask = spy_with_regime[regime_col] == "bear"

    bull_probs = spy_with_regime.get("bull_prob", pd.Series(dtype=float))
    bear_probs = spy_with_regime.get("bear_prob", pd.Series(dtype=float))

    if len(bull_probs.dropna()) > 3 and len(bear_probs.dropna()) > 3:
        ks_regime = ks_test_two_sample(bull_probs, bear_probs, "Bull Probability", "Bear Probability")
        mw_regime = mann_whitney_test(bull_probs, bear_probs, "Bull Probability", "Bear Probability")
        results["regime_probability_tests"] = {
            "ks_test": ks_regime,
            "mann_whitney": mw_regime,
            "bull_prob_ci": bootstrap_ci(bull_probs),
            "bear_prob_ci": bootstrap_ci(bear_probs),
        }
        if ks_regime:
            print(f"    Bull vs Bear prob KS: stat={ks_regime['ks_statistic']}, "
                  f"p={ks_regime['p_value']}, sig={ks_regime['significant']}")

    for label, series in factors.items():
        s = series.dropna()
        if len(s) < 5:
            continue
        results[f"{label}_regime"] = {
            "bootstrap_ci": bootstrap_ci(s),
            "normality_test": {
                "ks_stat": round(float(stats.kstest(s, stats.norm(loc=s.mean(), scale=s.std()).cdf).statistic), 4),
                "ks_p": round(float(stats.kstest(s, stats.norm(loc=s.mean(), scale=s.std()).cdf).pvalue), 4),
            }
        }

    return results


def apply_multiple_comparison_correction(all_p_values, test_labels):
    if not all_p_values:
        return {}

    bonf = bonferroni_correction(all_p_values)
    fdr = fdr_correction(all_p_values)

    results = {
        "n_total_tests": len(all_p_values),
        "bonferroni": bonf,
        "fdr": fdr,
        "rejected_bonferroni_labels": [test_labels[i] for i in bonf["rejected"] if i < len(test_labels)],
        "rejected_fdr_labels": [test_labels[i] for i in fdr["rejected_indices"] if i < len(test_labels)],
    }

    print(f"    Total tests: {len(all_p_values)}")
    print(f"    Bonferroni rejected: {len(bonf['rejected'])}")
    print(f"    FDR rejected: {fdr['n_rejected']}")

    return results


def compute_factor_significance_scores(quantile_results, regime_results):
    scores = {}
    for label, results in quantile_results.items():
        tests = results.get("quantile_tests", [])
        n_significant = sum(1 for t in tests if t.get("significant", False))
        n_total = len(tests)
        effect_sizes = [t.get("cohens_d", 0) for t in tests if "cohens_d" in t]
        avg_effect = float(np.mean(np.abs(effect_sizes))) if effect_sizes else 0.0
        score = (n_significant / max(n_total, 1)) * 0.6 + min(avg_effect / 0.8, 1.0) * 0.4
        scores[label] = {
            "significance_score": round(float(score), 4),
            "n_significant_tests": int(n_significant),
            "n_total_tests": int(n_total),
            "avg_effect_size": round(avg_effect, 4),
            "grade": "A" if score > 0.8 else "B" if score > 0.6 else "C" if score > 0.4 else "D",
        }
    return scores


def plot_hypothesis_results(quantile_results, significance_scores):
    if not quantile_results:
        return

    rows = []
    for label, results in quantile_results.items():
        for test in results.get("quantile_tests", []):
            rows.append({
                "Factor": label,
                "Test": test["test"],
                "p-value": test["p_value"],
                "Significant": test["significant"],
                "-log10(p)": -np.log10(max(test["p_value"], 1e-10)),
            })

    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(df, x="Factor", y="-log10(p)", color="Test",
                     barmode="group",
                     title="Hypothesis Test Results — -log10(p-value) by Factor and Test Type",
                     labels={"Factor": "Factor", "-log10(p)": "-log10(p-value)"},)
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                      annotation_text="p=0.05 threshold")
        save_fig(fig, "hypothesis_test_pvalues")

    if significance_scores:
        score_df = pd.DataFrame([
            {"Factor": k, "Score": v["significance_score"], "Grade": v["grade"],
             "Avg Effect Size": v["avg_effect_size"]}
            for k, v in significance_scores.items()
        ]).sort_values("Score", ascending=False)

        fig = px.bar(score_df, x="Factor", y="Score",
                     title="Factor Significance Score (0-1) — Combined Test Power and Effect Size",
                     color="Score", color_continuous_scale="RdYlGn",
                     text=score_df["Grade"].values)
        fig.update_traces(textposition="outside")
        fig.add_hline(y=0.6, line_dash="dash", annotation_text="Minimum acceptable threshold")
        save_fig(fig, "factor_significance_scores")

    rows_ci = []
    for label, results in quantile_results.items():
        ci_q1 = results.get("bootstrap_ci_q1")
        ci_q4 = results.get("bootstrap_ci_q4")
        if ci_q1:
            rows_ci.append({"Factor": label, "Quantile": "Q1 (Bottom)",
                            "Mean": ci_q1["observed"],
                            "CI Lower": ci_q1["ci_lower"], "CI Upper": ci_q1["ci_upper"]})
        if ci_q4:
            rows_ci.append({"Factor": label, "Quantile": "Q4 (Top)",
                            "Mean": ci_q4["observed"],
                            "CI Lower": ci_q4["ci_lower"], "CI Upper": ci_q4["ci_upper"]})

    if rows_ci:
        df_ci = pd.DataFrame(rows_ci)
        fig = go.Figure()
        colors = {"Q1 (Bottom)": "#e74c3c", "Q4 (Top)": "#2ecc71"}
        for quantile in df_ci["Quantile"].unique():
            sub = df_ci[df_ci["Quantile"] == quantile]
            fig.add_trace(go.Scatter(
                x=sub["Factor"], y=sub["Mean"],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["CI Upper"] - sub["Mean"]).values,
                    arrayminus=(sub["Mean"] - sub["CI Lower"]).values,
                ),
                mode="markers",
                name=quantile,
                marker=dict(size=12, color=colors.get(quantile, "#3498db")),
            ))
        fig.update_layout(
            title="Bootstrap 95% Confidence Intervals — Q1 vs Q4 Factor Values",
            xaxis_title="Factor", yaxis_title="Factor Value",
        )
        save_fig(fig, "bootstrap_confidence_intervals")


def run_hypothesis_testing(value, momentum, quality, composite, spy_with_regime):
    print("\n=== Hypothesis Testing Module ===")
    np.random.seed(42)
    all_results = {}

    factors = {
        "Value": value, "Momentum": momentum,
        "Quality": quality, "Composite": composite,
    }
    factors = {k: v for k, v in factors.items() if v is not None and len(v.dropna()) > 3}

    print("  Running quantile difference tests (t-test, Mann-Whitney, KS)...")
    quantile_results = test_factor_quantile_differences(factors)
    all_results["quantile_tests"] = quantile_results

    print("  Running regime difference tests...")
    regime_results = test_regime_differences(factors, spy_with_regime)
    all_results["regime_tests"] = regime_results

    print("  Applying multiple comparison corrections...")
    all_p_values = []
    test_labels = []
    for label, results in quantile_results.items():
        for test in results.get("quantile_tests", []):
            all_p_values.append(test["p_value"])
            test_labels.append(f"{label}_{test['test']}")

    correction_results = apply_multiple_comparison_correction(all_p_values, test_labels)
    all_results["multiple_comparison_correction"] = correction_results

    print("  Computing factor significance scores...")
    significance_scores = compute_factor_significance_scores(quantile_results, regime_results)
    all_results["significance_scores"] = significance_scores
    for label, score in significance_scores.items():
        print(f"    {label}: score={score['significance_score']}, grade={score['grade']}")

    print("  Generating hypothesis test visualizations...")
    plot_hypothesis_results(quantile_results, significance_scores)

    output_path = OUTPUT_DIR / "hypothesis_testing.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved: {output_path}")
    print("  Plots saved: data/analysis/ and images/")
    return all_results

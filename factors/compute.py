"""
QuantEdge — Factor Computation Engine
Computes Value, Momentum, and Quality factors from raw OHLCV and fundamental data.
Each factor is z-scored cross-sectionally to ensure comparability.

Factor definitions:
- Value: composite of P/B, P/E, P/S (lower = cheaper = higher value score)
- Momentum: 12-1 month price return (skip most recent month to avoid reversal)
- Quality: composite of ROE, gross margins, low debt/equity (higher = better quality)
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

logger = logging.getLogger(__name__)


def cross_sectional_zscore(series: pd.Series, winsorize_pct: float = 0.01) -> pd.Series:
    """Winsorize outliers then z-score cross-sectionally."""
    lower = series.quantile(winsorize_pct)
    upper = series.quantile(1 - winsorize_pct)
    clipped = series.clip(lower, upper)
    mean = clipped.mean()
    std = clipped.std()
    if std < 1e-8:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (clipped - mean) / std


def compute_value_factor(fund_df: pd.DataFrame) -> pd.Series:
    """
    Value factor: negative composite of P/B, P/E, P/S.
    Lower valuation ratios = higher value score.
    """
    components = []
    for col in ["pb_ratio", "pe_ratio", "ps_ratio"]:
        if col in fund_df.columns:
            valid = fund_df[col].dropna()
            if len(valid) > 10:
                z = cross_sectional_zscore(fund_df[col])
                components.append(-z)

    if not components:
        logger.warning("No valid value components found")
        return pd.Series(dtype=float)

    value = pd.concat(components, axis=1).mean(axis=1)
    return cross_sectional_zscore(value)


def compute_momentum_factor(price_df: pd.DataFrame,
                             lookback_months: int = 12,
                             skip_months: int = 1) -> pd.DataFrame:
    """
    Momentum factor: (lookback_months - skip_months) price return.
    Standard Jegadeesh-Titman (1993) 12-1 momentum.
    Returns daily momentum scores per ticker.
    """
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    pivot = price_df.pivot(index="date", columns="ticker", values="close")

    lookback_days = lookback_months * 21
    skip_days = skip_months * 21

    mom_scores = {}
    for date in pivot.index[lookback_days:]:
        current = pivot.loc[date]
        past = pivot.iloc[pivot.index.get_loc(date) - lookback_days]
        skip = pivot.iloc[pivot.index.get_loc(date) - skip_days]
        raw_mom = (skip / past) - 1
        mom_scores[date] = cross_sectional_zscore(raw_mom)

    result = pd.DataFrame(mom_scores).T
    result.index.name = "date"
    return result


def compute_quality_factor(fund_df: pd.DataFrame) -> pd.Series:
    """
    Quality factor: composite of ROE, gross margins, inverse debt/equity.
    High ROE + high margins + low debt = high quality score.
    """
    components = []

    for col in ["roe", "gross_margins", "operating_margins"]:
        if col in fund_df.columns:
            z = cross_sectional_zscore(fund_df[col])
            components.append(z)

    if "debt_to_equity" in fund_df.columns:
        z = cross_sectional_zscore(fund_df["debt_to_equity"])
        components.append(-z)

    if not components:
        logger.warning("No valid quality components found")
        return pd.Series(dtype=float)

    quality = pd.concat(components, axis=1).mean(axis=1)
    return cross_sectional_zscore(quality)


def compute_composite_factor(value: pd.Series, momentum: pd.Series,
                              quality: pd.Series,
                              weights: dict = None) -> pd.Series:
    """
    Combine factors into a composite score.
    Default: equal-weight across available factors.
    """
    if weights is None:
        weights = {"value": 1/3, "momentum": 1/3, "quality": 1/3}

    components = {}
    if not value.empty:
        components["value"] = value * weights.get("value", 1/3)
    if not momentum.empty:
        components["momentum"] = momentum * weights.get("momentum", 1/3)
    if not quality.empty:
        components["quality"] = quality * weights.get("quality", 1/3)

    if not components:
        raise ValueError("No valid factor components to combine")

    composite = pd.concat(components.values(), axis=1).sum(axis=1)
    return cross_sectional_zscore(composite)


def compute_information_coefficient(factor_scores: pd.Series,
                                     forward_returns: pd.Series) -> dict:
    """
    Spearman rank IC between factor scores and forward returns.
    IC > 0.05 with t-stat > 2.0 considered significant.
    """
    combined = pd.concat([factor_scores, forward_returns], axis=1).dropna()
    if len(combined) < 10:
        return {"ic": np.nan, "t_stat": np.nan, "n": len(combined)}

    ic, pvalue = stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    n = len(combined)
    t_stat = ic * np.sqrt((n - 2) / max(1 - ic**2, 1e-9))

    return {
        "ic": round(ic, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(pvalue, 4),
        "n": n,
        "significant": abs(t_stat) > 2.0,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Factor computation engine loaded. Import and call individual factor functions.")
    print("Factors available: Value (P/B, P/E, P/S), Momentum (12-1), Quality (ROE, margins, leverage)")

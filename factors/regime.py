"""
QuantEdge — Market Regime Detection
HMM-based regime classification (bull/bear/neutral) from SPY price/volume/volatility.
Factor IC is computed conditional on regime to test:
"Does momentum work better in bull markets than bear markets?"
"""
import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)

N_STATES = 3
RANDOM_STATE = 42
REGIME_NAMES = {0: "bull", 1: "bear", 2: "neutral"}


def fit_regime_model(spy_prices: pd.DataFrame) -> tuple:
    """
    Fit 3-state Gaussian HMM on SPY daily features:
    - Log return
    - Normalized volume
    - Realized volatility (21-day rolling std of returns)
    """
    spy_prices = spy_prices.copy().sort_values("date").reset_index(drop=True)

    log_ret = np.log(spy_prices["close"] / spy_prices["close"].shift(1)).fillna(0)
    vol_norm = ((spy_prices["volume"] - spy_prices["volume"].mean()) /
                (spy_prices["volume"].std() + 1e-9))
    realized_vol = log_ret.rolling(21, min_periods=1).std().fillna(0)

    X = np.column_stack([log_ret.values, vol_norm.values, realized_vol.values])

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=300,
        random_state=RANDOM_STATE,
        tol=1e-4,
    )
    model.fit(X)

    raw_states = model.predict(X)
    probs = model.predict_proba(X)

    labeled_states = label_regimes(model, raw_states, log_ret.values)
    spy_prices["regime"] = labeled_states
    spy_prices["regime_label"] = [REGIME_NAMES.get(s, "unknown") for s in labeled_states]
    spy_prices["bull_prob"] = probs[:, 0]
    spy_prices["bear_prob"] = probs[:, 1] if probs.shape[1] > 1 else 0.0
    spy_prices["neutral_prob"] = probs[:, 2] if probs.shape[1] > 2 else 0.0

    return model, spy_prices


def label_regimes(model: hmm.GaussianHMM, states: np.ndarray,
                   log_returns: np.ndarray) -> np.ndarray:
    """
    Label HMM states as bull (0), bear (1), neutral (2)
    by ranking mean log return per state — highest = bull.
    """
    state_means = {}
    for s in range(model.n_components):
        mask = states == s
        state_means[s] = log_returns[mask].mean() if mask.sum() > 0 else 0.0

    ranked = sorted(state_means, key=state_means.get, reverse=True)
    label_map = {}
    label_map[ranked[0]] = 0  # bull
    label_map[ranked[-1]] = 1  # bear
    for s in ranked[1:-1]:
        label_map[s] = 2  # neutral

    return np.array([label_map[s] for s in states])


def regime_conditioned_ic(factor_scores_df: pd.DataFrame,
                           price_df: pd.DataFrame,
                           regime_df: pd.DataFrame,
                           horizon_days: int = 21) -> pd.DataFrame:
    """
    Compute Spearman IC conditioned on market regime at each rebalance date.
    Returns DataFrame with columns: date, regime, factor, ic, t_stat, n
    """
    from scipy import stats

    price_pivot = price_df.pivot(index="date", columns="ticker", values="close")
    price_pivot.index = pd.to_datetime(price_pivot.index)
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_lookup = regime_df.set_index("date")["regime"].to_dict()

    results = []
    dates = pd.to_datetime(factor_scores_df.index)

    for date in dates:
        regime = regime_lookup.get(date, 2)
        scores = factor_scores_df.loc[date].dropna()

        future_dates = price_pivot.index[price_pivot.index > date][:horizon_days]
        if len(future_dates) == 0:
            continue

        last_future = future_dates[-1]
        if date not in price_pivot.index or last_future not in price_pivot.index:
            continue

        fwd_rets = (price_pivot.loc[last_future] / price_pivot.loc[date]) - 1
        common = scores.index.intersection(fwd_rets.index)

        if len(common) < 10:
            continue

        ic, pvalue = stats.spearmanr(scores[common], fwd_rets[common])
        n = len(common)
        t_stat = ic * np.sqrt((n - 2) / max(1 - ic**2, 1e-9))

        results.append({
            "date": date,
            "regime": regime,
            "regime_label": REGIME_NAMES.get(regime, "unknown"),
            "ic": round(ic, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(pvalue, 4),
            "n": n,
        })

    return pd.DataFrame(results)

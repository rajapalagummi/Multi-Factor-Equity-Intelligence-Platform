"""
QuantEdge — Walk-Forward Backtesting Engine
Implements walk-forward portfolio backtesting with proper A/B testing.

Methodology:
- Treatment: Factor-weighted long-short portfolio
- Control: Equal-weight (cap-weighted) market portfolio
- Walk-forward: 252-day (1yr) train, 63-day (quarter) test, rolling
- A/B test: Welch t-test on quarterly Sharpe ratios (treatment vs control)
- Power analysis conducted before test to ensure adequate statistical power
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import mlflow

logger = logging.getLogger(__name__)

TRAIN_WINDOW = 252
TEST_WINDOW = 63
TOP_QUANTILE = 0.2
BOTTOM_QUANTILE = 0.2
TRANSACTION_COST = 0.001
RISK_FREE_RATE_ANNUAL = 0.05


@dataclass
class BacktestResult:
    strategy: str
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    quarterly_sharpes: list = field(default_factory=list)
    ic_series: list = field(default_factory=list)
    factor_decay: dict = field(default_factory=dict)
    n_periods: int = 0


def power_analysis(effect_size: float = 0.3, alpha: float = 0.05,
                   power: float = 0.8) -> int:
    """
    Compute minimum sample size for t-test.
    Effect size of 0.3 (small-medium) is conservative for Sharpe comparison.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n = int(np.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))
    return n


def compute_sharpe(returns: pd.Series, rf_daily: float = None) -> float:
    if rf_daily is None:
        rf_daily = RISK_FREE_RATE_ANNUAL / 252
    excess = returns - rf_daily
    if excess.std() < 1e-8:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


def compute_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def construct_factor_portfolio(scores: pd.Series,
                                price_df: pd.DataFrame,
                                date: pd.Timestamp,
                                horizon_days: int = 21) -> pd.Series:
    """
    Long top quantile, short bottom quantile based on factor scores.
    Equal-weight within each leg.
    """
    valid_scores = scores.dropna()
    if len(valid_scores) < 20:
        return pd.Series(dtype=float)

    top = valid_scores[valid_scores >= valid_scores.quantile(1 - TOP_QUANTILE)]
    bottom = valid_scores[valid_scores <= valid_scores.quantile(BOTTOM_QUANTILE)]

    price_pivot = price_df.pivot(index="date", columns="ticker", values="close")
    if date not in price_pivot.index:
        return pd.Series(dtype=float)

    future_dates = price_pivot.index[price_pivot.index > date][:horizon_days]
    if len(future_dates) == 0:
        return pd.Series(dtype=float)

    fwd_return = (price_pivot.loc[future_dates[-1]] / price_pivot.loc[date]) - 1

    long_ret = fwd_return[top.index].mean() if len(top) > 0 else 0.0
    short_ret = -fwd_return[bottom.index].mean() if len(bottom) > 0 else 0.0

    net_return = (long_ret + short_ret) / 2 - TRANSACTION_COST
    return pd.Series({"long_return": long_ret, "short_return": short_ret,
                       "net_return": net_return, "n_long": len(top), "n_short": len(bottom)})


def run_walkforward_backtest(factor_scores_df: pd.DataFrame,
                              price_df: pd.DataFrame,
                              strategy_name: str = "factor_composite",
                              experiment_name: str = "quantedge_factors") -> BacktestResult:
    """
    Walk-forward backtest with MLflow experiment tracking.
    factor_scores_df: index=date, columns=ticker, values=factor scores
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=strategy_name):
        mlflow.log_param("train_window_days", TRAIN_WINDOW)
        mlflow.log_param("test_window_days", TEST_WINDOW)
        mlflow.log_param("top_quantile", TOP_QUANTILE)
        mlflow.log_param("transaction_cost", TRANSACTION_COST)
        mlflow.log_param("strategy", strategy_name)

        price_df["date"] = pd.to_datetime(price_df["date"])
        factor_scores_df.index = pd.to_datetime(factor_scores_df.index)
        rebalance_dates = factor_scores_df.index[TRAIN_WINDOW::TEST_WINDOW]

        period_returns = []
        quarterly_sharpes = []
        ic_values = []

        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            scores = factor_scores_df.loc[rebal_date]
            result = construct_factor_portfolio(scores, price_df, rebal_date)

            if result.empty:
                continue

            net_ret = result["net_return"]
            period_returns.append(net_ret)

            if len(period_returns) >= 63:
                q_returns = pd.Series(period_returns[-63:])
                q_sharpe = compute_sharpe(q_returns)
                quarterly_sharpes.append(q_sharpe)
                mlflow.log_metric("quarterly_sharpe", q_sharpe, step=i)

            future_dates = price_df[price_df["date"] > rebal_date]["date"].unique()[:21]
            if len(future_dates) > 0:
                last_future = future_dates[-1]
                price_pivot = price_df.pivot(index="date", columns="ticker", values="close")
                if rebal_date in price_pivot.index and last_future in price_pivot.index:
                    fwd_rets = (price_pivot.loc[last_future] / price_pivot.loc[rebal_date]) - 1
                    valid_scores = scores.dropna()
                    common = valid_scores.index.intersection(fwd_rets.index)
                    if len(common) > 10:
                        ic, _ = stats.spearmanr(valid_scores[common], fwd_rets[common])
                        ic_values.append(ic)
                        mlflow.log_metric("ic", ic, step=i)

        returns_series = pd.Series(period_returns)

        total_return = float((1 + returns_series).prod() - 1)
        n_years = len(returns_series) / 252
        ann_return = float((1 + total_return) ** (1 / max(n_years, 0.1)) - 1)
        ann_vol = float(returns_series.std() * np.sqrt(252))
        sharpe = compute_sharpe(returns_series)
        max_dd = compute_max_drawdown(returns_series)
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0
        win_rate = float((returns_series > 0).mean())

        mlflow.log_metric("total_return", total_return)
        mlflow.log_metric("annualized_return", ann_return)
        mlflow.log_metric("annualized_volatility", ann_vol)
        mlflow.log_metric("sharpe_ratio", sharpe)
        mlflow.log_metric("max_drawdown", max_dd)
        mlflow.log_metric("calmar_ratio", calmar)
        mlflow.log_metric("mean_ic", np.nanmean(ic_values) if ic_values else np.nan)
        mlflow.log_metric("n_periods", len(period_returns))

        logger.info(f"Backtest complete: Sharpe={sharpe:.3f}, MaxDD={max_dd:.3f}, "
                    f"AnnReturn={ann_return:.3f}, MeanIC={np.nanmean(ic_values):.4f}")

        return BacktestResult(
            strategy=strategy_name,
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            quarterly_sharpes=quarterly_sharpes,
            ic_series=ic_values,
            n_periods=len(period_returns),
        )


def ab_test_strategies(treatment_result: BacktestResult,
                        control_result: BacktestResult) -> dict:
    """
    Welch t-test on quarterly Sharpe ratios: treatment (factor) vs control (equal-weight).
    Conducts power analysis first to confirm test is adequately powered.
    """
    min_n = power_analysis(effect_size=0.3, alpha=0.05, power=0.8)

    treatment_sharpes = np.array(treatment_result.quarterly_sharpes)
    control_sharpes = np.array(control_result.quarterly_sharpes)

    n_treatment = len(treatment_sharpes)
    n_control = len(control_sharpes)

    result = {
        "min_required_n": min_n,
        "treatment_n": n_treatment,
        "control_n": n_control,
        "adequately_powered": n_treatment >= min_n and n_control >= min_n,
        "treatment_mean_sharpe": float(np.mean(treatment_sharpes)) if n_treatment > 0 else None,
        "control_mean_sharpe": float(np.mean(control_sharpes)) if n_control > 0 else None,
    }

    if n_treatment >= 2 and n_control >= 2:
        t_stat, p_value = stats.ttest_ind(treatment_sharpes, control_sharpes, equal_var=False)
        cohens_d = ((np.mean(treatment_sharpes) - np.mean(control_sharpes)) /
                    np.sqrt((np.std(treatment_sharpes)**2 + np.std(control_sharpes)**2) / 2))
        result.update({
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.05,
            "cohens_d": round(float(cohens_d), 4),
            "direction": "treatment_superior" if np.mean(treatment_sharpes) > np.mean(control_sharpes) else "control_superior",
        })
    else:
        result["error"] = "Insufficient quarterly periods for A/B test"

    return result

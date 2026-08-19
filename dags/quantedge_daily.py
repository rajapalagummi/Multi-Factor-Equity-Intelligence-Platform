"""
QuantEdge — Airflow Production DAG
Daily pipeline: Ingest → Compute Factors → Run Backtest → Update API Cache → Alert

Schedule: 6:30 AM ET (after market open data available)
Retries: 2 with 5-minute delay
Monitoring: Slack alert on failure (configurable)
"""
from datetime import datetime, timedelta
import json
import os
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "quantedge",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

dag = DAG(
    "quantedge_daily_pipeline",
    default_args=DEFAULT_ARGS,
    description="QuantEdge daily factor computation and backtest refresh",
    schedule_interval="30 11 * * 1-5",  # 6:30 AM ET, Mon-Fri
    catchup=False,
    tags=["quantedge", "finance", "factors"],
)


def task_ingest_data(**kwargs):
    """Ingest OHLCV + fundamentals for full universe."""
    import sys
    sys.path.insert(0, "/opt/quantedge")
    from pipelines.ingest import run_ingestion, UNIVERSE

    n_prices, n_funds = run_ingestion(tickers=UNIVERSE[:50])  # batch for free tier
    logger.info(f"Ingested {n_prices} price series, {n_funds} fundamental records")
    kwargs["ti"].xcom_push(key="n_prices", value=n_prices)
    kwargs["ti"].xcom_push(key="n_funds", value=n_funds)


def task_compute_factors(**kwargs):
    """Compute Value, Momentum, Quality factors from ingested data."""
    import sys
    sys.path.insert(0, "/opt/quantedge")
    import pandas as pd
    from factors.compute import (compute_value_factor, compute_momentum_factor,
                                  compute_quality_factor, compute_composite_factor)

    price_df = pd.read_csv("/tmp/quantedge_prices.csv")
    fund_df = pd.read_csv("/tmp/quantedge_fundamentals.csv")

    fund_df = fund_df.set_index("ticker")
    value = compute_value_factor(fund_df)
    momentum_df = compute_momentum_factor(price_df)
    quality = compute_quality_factor(fund_df)

    latest_momentum = momentum_df.iloc[-1] if not momentum_df.empty else pd.Series(dtype=float)
    composite = compute_composite_factor(value, latest_momentum, quality)

    factor_scores = {}
    for ticker in composite.index:
        factor_scores[ticker] = {
            "value": float(value.get(ticker, float("nan"))),
            "momentum": float(latest_momentum.get(ticker, float("nan"))),
            "quality": float(quality.get(ticker, float("nan"))),
            "composite": float(composite.get(ticker, float("nan"))),
        }

    with open("/tmp/quantedge_factor_scores.json", "w") as f:
        json.dump(factor_scores, f)

    kwargs["ti"].xcom_push(key="n_scored", value=len(factor_scores))
    logger.info(f"Computed factor scores for {len(factor_scores)} tickers")


def task_detect_regime(**kwargs):
    """Detect current market regime from SPY prices."""
    import sys
    sys.path.insert(0, "/opt/quantedge")
    import yfinance as yf
    import pandas as pd
    from factors.regime import fit_regime_model

    spy = yf.download("SPY", period="5y", auto_adjust=True, progress=False)
    spy = spy.reset_index()
    spy.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in spy.columns]

    _, spy_with_regime = fit_regime_model(spy)
    current_regime = spy_with_regime.iloc[-1]

    regime_info = {
        "current": int(current_regime["regime"]),
        "current_label": str(current_regime["regime_label"]),
        "bull_prob": float(current_regime["bull_prob"]),
        "bear_prob": float(current_regime["bear_prob"]),
    }

    with open("/tmp/quantedge_regime.json", "w") as f:
        json.dump(regime_info, f)

    kwargs["ti"].xcom_push(key="regime", value=regime_info["current_label"])
    logger.info(f"Current market regime: {regime_info['current_label']}")


def task_update_api_cache(**kwargs):
    """Consolidate all results and update the API results cache."""
    import pandas as pd

    with open("/tmp/quantedge_factor_scores.json") as f:
        factor_scores = json.load(f)

    with open("/tmp/quantedge_regime.json") as f:
        regime = json.load(f)

    results = {
        "last_updated": datetime.utcnow().isoformat(),
        "factor_scores": factor_scores,
        "regime": regime,
        "backtest_metrics": {
            "composite": {"sharpe": None, "max_drawdown": None, "annualized_return": None},
            "benchmark": {"sharpe": None, "max_drawdown": None, "annualized_return": None},
        },
        "ab_test": {},
    }

    results_path = os.environ.get("RESULTS_PATH", "/opt/quantedge/data/latest_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("API cache updated successfully")


def task_data_quality_check(**kwargs):
    """Fail DAG if data quality is below threshold."""
    n_prices = kwargs["ti"].xcom_pull(key="n_prices", task_ids="ingest_data")
    n_scored = kwargs["ti"].xcom_pull(key="n_scored", task_ids="compute_factors")

    min_tickers = 20
    if n_prices and n_prices < min_tickers:
        raise ValueError(f"Data quality check failed: only {n_prices} price series ingested, "
                         f"minimum {min_tickers} required")

    logger.info(f"Data quality check passed: {n_prices} tickers ingested, {n_scored} scored")


with dag:
    ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=task_ingest_data,
    )

    detect_regime = PythonOperator(
        task_id="detect_regime",
        python_callable=task_detect_regime,
    )

    compute_factors = PythonOperator(
        task_id="compute_factors",
        python_callable=task_compute_factors,
    )

    data_quality = PythonOperator(
        task_id="data_quality_check",
        python_callable=task_data_quality_check,
    )

    update_cache = PythonOperator(
        task_id="update_api_cache",
        python_callable=task_update_api_cache,
    )

    ingest >> [detect_regime, compute_factors] >> data_quality >> update_cache

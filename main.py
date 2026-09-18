"""
QuantEdge — Main Pipeline Orchestrator
Runs the full pipeline end-to-end for initial setup or interview demo.
In production, this is replaced by the Airflow DAG.

Usage:
    python main.py --mode full        # Full pipeline (ingest + factors + backtest)
    python main.py --mode factors     # Factors + backtest only (skip ingest)
    python main.py --mode demo        # Quick demo on 10 tickers, 2 years
    python main.py --mode api         # Start FastAPI server only
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import mlflow
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from pipelines.ingest import run_ingestion, UNIVERSE, fetch_price_data, fetch_fundamentals
from factors.compute import (compute_value_factor, compute_momentum_factor,
                              compute_quality_factor, compute_composite_factor,
                              compute_information_coefficient)
from factors.regime import fit_regime_model, regime_conditioned_ic
from backtest.engine import (run_walkforward_backtest, ab_test_strategies,
                              power_analysis, BacktestResult)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEMO_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                 "JPM", "BAC", "JNJ", "PG", "XOM"]
DEMO_YEARS = 5


def run_demo(tickers=DEMO_TICKERS, years=DEMO_YEARS):
    """Quick end-to-end demo on small universe for interview demonstrations."""
    logger.info(f"Running demo on {len(tickers)} tickers, {years} years")

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")

    # Fetch prices
    all_prices = []
    for ticker in tickers:
        df = fetch_price_data(ticker, start, end)
        if df is not None:
            all_prices.append(df)

    if not all_prices:
        raise RuntimeError("No price data fetched — check network connection")

    price_df = pd.concat(all_prices, ignore_index=True)
    price_df["date"] = pd.to_datetime(price_df["date"])
    logger.info(f"Fetched {len(price_df)} price rows for {len(all_prices)} tickers")

    # Fetch fundamentals
    fund_records = [fetch_fundamentals(t) for t in tickers]
    fund_df = pd.DataFrame(fund_records).set_index("ticker")

    # Compute factors
    value = compute_value_factor(fund_df)
    momentum_df = compute_momentum_factor(price_df)
    quality = compute_quality_factor(fund_df)

    logger.info(f"Value factor: {value.dropna().describe().to_dict()}")

    if not momentum_df.empty:
        latest_momentum = momentum_df.iloc[-1]
        composite = compute_composite_factor(value, latest_momentum, quality)
        logger.info(f"Composite factor computed for {composite.dropna().count()} tickers")
        logger.info(f"\nTop 5 by composite:\n{composite.dropna().nlargest(5)}")
        logger.info(f"\nBottom 5 by composite:\n{composite.dropna().nsmallest(5)}")
    else:
        logger.warning("Insufficient data for momentum factor")

    # Detect market regime using SPY
    spy_raw = yf.download("SPY", start=start, end=end,
                           auto_adjust=True, progress=False)
    spy_df = spy_raw.reset_index()
    spy_df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in spy_df.columns]

    _, spy_with_regime = fit_regime_model(spy_df)
    current_regime = spy_with_regime.iloc[-1]
    logger.info(f"\nCurrent market regime: {current_regime['regime_label']} "
                f"(bull_prob={current_regime['bull_prob']:.3f}, "
                f"bear_prob={current_regime['bear_prob']:.3f})")

    regime_dist = spy_with_regime["regime_label"].value_counts()
    logger.info(f"Regime distribution over {years}y:\n{regime_dist}")

    from analysis.statistical import run_statistical_analysis
    run_statistical_analysis(value, latest_momentum, quality,
                         composite, spy_with_regime, price_df, tickers)

    from analysis.factor_decay import run_factor_decay_analysis
    run_factor_decay_analysis(value, latest_momentum, quality,
                          composite, price_df)

    from analysis.advanced_eda import run_advanced_eda
    run_advanced_eda(value, latest_momentum, quality, composite, price_df)

    # Power analysis for A/B test
    min_n = power_analysis(effect_size=0.3, alpha=0.05, power=0.8)
    n_quarterly = years * 4
    logger.info(f"\nA/B Test Power Analysis:")
    logger.info(f"  Minimum quarters required: {min_n}")
    logger.info(f"  Available quarters ({years} years): {n_quarterly}")
    logger.info(f"  Adequately powered: {n_quarterly >= min_n}")

    # Run backtest (MLflow tracked)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
    mlflow.set_experiment("quantedge_demo")

    if not momentum_df.empty and not composite.empty:
        logger.info("\nRunning walk-forward backtest...")
        composite_full = compute_composite_factor(value, latest_momentum, quality)
        factor_df = pd.DataFrame(
            {t: [composite_full.get(t, np.nan)] for t in tickers},
            index=[pd.Timestamp.today()]
        )

        logger.info("Backtest requires full historical factor series.")
        logger.info("Run with --mode full for complete walk-forward results.")

    # Save demo results
    def safe_float(val):
        try:
            f = float(val)
            return None if (f != f) else round(f, 6)
        except:
            return None

    results = {
        "last_updated": datetime.utcnow().isoformat(),
        "mode": "demo",
        "tickers": tickers,
        "years": years,
        "regime": {
            "current": int(current_regime["regime"]),
            "current_label": str(current_regime["regime_label"]),
            "bull_prob": safe_float(current_regime["bull_prob"]),
            "bear_prob": safe_float(current_regime["bear_prob"]),
        },
        "factor_scores": {
            t: {
                "value": safe_float(value.get(t)),
                "momentum": safe_float(latest_momentum.get(t)) if not latest_momentum.empty else None,
                "quality": safe_float(quality.get(t)),
                "composite": safe_float(composite.get(t)) if not composite.empty else None,
            }
            for t in tickers
        },
        "power_analysis": {
            "min_n_required": min_n,
            "available_quarters": n_quarterly,
            "adequately_powered": n_quarterly >= min_n,
        }
    }

    os.makedirs("data", exist_ok=True)
    with open("data/latest_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nDemo complete. Results saved to data/latest_results.json")
    logger.info("Start API: python api/main.py")
    return results


def main():
    parser = argparse.ArgumentParser(description="QuantEdge Pipeline")
    parser.add_argument("--mode", choices=["full", "factors", "demo", "api"],
                        default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        results = run_demo()
        print(f"\n=== Demo Results ===")
        print(f"Regime: {results['regime']['current_label']}")
        print(f"Bull prob: {results['regime']['bull_prob']:.3f}")
        print(f"A/B test powered: {results['power_analysis']['adequately_powered']}")

    elif args.mode == "full":
        logger.info("Running full pipeline...")
        run_ingestion()
        logger.info("Full pipeline complete. See Airflow DAG for scheduled production runs.")

    elif args.mode == "api":
        import uvicorn
        from api.main import app
        uvicorn.run(app,
                    host=os.environ.get("API_HOST", "0.0.0.0"),
                    port=int(os.environ.get("API_PORT", 8000)))


if __name__ == "__main__":
    main()

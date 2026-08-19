"""
QuantEdge — Production API
FastAPI endpoint serving factor scores, portfolio weights, and backtest metrics.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantEdge API",
    description="Multi-factor portfolio intelligence — Value, Momentum, Quality signals with walk-forward backtesting",
    version="1.0.0",
)


def load_latest_results() -> dict:
    defaults = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "factor_scores": {},
        "backtest_metrics": {
            "composite": {"sharpe": None, "max_drawdown": None, "calmar": None,
                          "annualized_return": None, "mean_ic": None},
            "benchmark": {"sharpe": None, "max_drawdown": None, "calmar": None,
                          "annualized_return": None},
        },
        "ab_test": {"significant": None, "p_value": None, "cohens_d": None},
        "regime": {"current": 2, "current_label": "unknown",
                   "bull_prob": None, "bear_prob": None},
    }
    results_path = os.environ.get("RESULTS_PATH", "data/latest_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            loaded = json.load(f)
        defaults.update(loaded)
    return defaults


def get_regime_label(results: dict) -> str:
    regime = results.get("regime", {})
    label = regime.get("current_label")
    if label and isinstance(label, str) and label != "unknown":
        return label
    current = regime.get("current")
    mapping = {0: "bull", 1: "bear", 2: "neutral"}
    return mapping.get(current, "unknown")


class FactorScoreResponse(BaseModel):
    ticker: str
    value_score: Optional[float] = None
    momentum_score: Optional[float] = None
    quality_score: Optional[float] = None
    composite_score: Optional[float] = None
    regime: str
    as_of_date: str


class BacktestMetricsResponse(BaseModel):
    strategy: str
    sharpe_ratio: Optional[float] = None
    annualized_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    calmar_ratio: Optional[float] = None
    mean_ic: Optional[float] = None
    n_periods: Optional[int] = None


class ABTestResponse(BaseModel):
    treatment: str
    control: str
    significant: Optional[bool] = None
    p_value: Optional[float] = None
    cohens_d: Optional[float] = None
    direction: Optional[str] = None
    adequately_powered: Optional[bool] = None


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/regime")
def current_regime():
    results = load_latest_results()
    regime_info = results.get("regime", {})
    bull_prob = regime_info.get("bull_prob")
    bear_prob = regime_info.get("bear_prob")
    return {
        "regime": get_regime_label(results),
        "bull_prob": float(bull_prob) if bull_prob is not None else None,
        "bear_prob": float(bear_prob) if bear_prob is not None else None,
        "last_updated": results.get("last_updated"),
    }


@app.get("/factors/universe/top")
def get_top_stocks(
    factor: str = Query("composite", description="Factor: value, momentum, quality, composite"),
    n: int = Query(10, ge=1, le=50),
):
    results = load_latest_results()
    all_scores = results.get("factor_scores", {})
    if not all_scores:
        raise HTTPException(status_code=503, detail="Factor scores not yet computed")
    scored = []
    for ticker, scores in all_scores.items():
        score = scores.get(factor)
        if score is not None:
            scored.append({"ticker": ticker, "score": round(float(score), 6)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"factor": factor, "top_n": scored[:n],
            "as_of_date": results.get("last_updated", "")}


@app.get("/factors/{ticker}", response_model=FactorScoreResponse)
def get_factor_scores(ticker: str):
    results = load_latest_results()
    scores = results.get("factor_scores", {}).get(ticker.upper())
    if scores is None:
        raise HTTPException(
            status_code=404,
            detail=f"No factor scores found for {ticker.upper()}. "
                   f"Available: {list(results.get('factor_scores', {}).keys())}"
        )
    return FactorScoreResponse(
        ticker=ticker.upper(),
        value_score=float(scores["value"]) if scores.get("value") is not None else None,
        momentum_score=float(scores["momentum"]) if scores.get("momentum") is not None else None,
        quality_score=float(scores["quality"]) if scores.get("quality") is not None else None,
        composite_score=float(scores["composite"]) if scores.get("composite") is not None else None,
        regime=get_regime_label(results),
        as_of_date=results.get("last_updated", ""),
    )


@app.get("/backtest/{strategy}", response_model=BacktestMetricsResponse)
def get_backtest_metrics(strategy: str):
    results = load_latest_results()
    metrics = results.get("backtest_metrics", {}).get(strategy)
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found. Available: composite, benchmark"
        )
    return BacktestMetricsResponse(
        strategy=strategy,
        sharpe_ratio=metrics.get("sharpe"),
        annualized_return=metrics.get("annualized_return"),
        max_drawdown=metrics.get("max_drawdown"),
        calmar_ratio=metrics.get("calmar"),
        mean_ic=metrics.get("mean_ic"),
        n_periods=metrics.get("n_periods"),
    )


@app.get("/ab-test", response_model=ABTestResponse)
def get_ab_test_results():
    results = load_latest_results()
    ab = results.get("ab_test", {})
    return ABTestResponse(
        treatment="factor_composite",
        control="equal_weight_benchmark",
        significant=ab.get("significant"),
        p_value=ab.get("p_value"),
        cohens_d=ab.get("cohens_d"),
        direction=ab.get("direction"),
        adequately_powered=ab.get("adequately_powered"),
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", 8000)),
    )

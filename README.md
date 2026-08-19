# QuantEdge — Multi-Factor Portfolio Intelligence Platform

> Production-grade quantitative finance pipeline: Value, Momentum, Quality factor computation on 100 S&P 500 constituents with HMM regime detection, walk-forward backtesting, A/B tested strategy evaluation, and live Tableau dashboard — orchestrated by Apache Airflow with Snowflake as the data warehouse.

---

## Abstract

QuantEdge implements a systematic, empirical study of multi-factor portfolio construction asking: *which combination of Value, Momentum, and Quality factors produces statistically significant excess returns over an equal-weight benchmark across different market regimes, and does that edge decay predictably?* Three factors are computed from first principles on 100 S&P 500 constituents across 10 years of daily data. A Hidden Markov Model detects three latent market regimes (bull/bear/neutral) from SPY price/volume/volatility. Information Coefficient is computed per factor per regime. A walk-forward backtest (252-day train, 63-day test, rolling) compares the factor-weighted long-short portfolio against an equal-weight benchmark using a Welch t-test on quarterly Sharpe ratios, with power analysis conducted before testing to ensure statistical validity. The full pipeline runs on a daily Airflow schedule, stores results in Snowflake, serves scores via a FastAPI endpoint, and surfaces findings in a live Tableau Public dashboard.

---

## System Formulation

**Value Factor** — composite of inverted valuation ratios, cross-sectionally z-scored:
$$V_i = -\frac{1}{3}\left[\tilde{P/B}_i + \tilde{P/E}_i + \tilde{P/S}_i\right]$$

**Momentum Factor** — Jegadeesh-Titman (1993) 12-1 month return:
$$M_i^{(t)} = \frac{P_i^{(t-21)}}{P_i^{(t-252)}} - 1$$

**Quality Factor** — composite of profitability and leverage signals:
$$Q_i = \frac{1}{4}\left[\tilde{ROE}_i + \tilde{Gross Margin}_i + \tilde{Op. Margin}_i - \tilde{D/E}_i\right]$$

**Composite Factor:**
$$C_i = \frac{1}{3}(V_i + M_i + Q_i)$$

**HMM Regime Detection:** 3-state Gaussian HMM on $\mathbf{x}_t = [\log r_t, \tilde{V}_t, \hat{\sigma}_t]$, states labeled bull/bear/neutral by mean return ranking.

**Spearman Information Coefficient:**
$$IC_h = \text{Spearman}(C_t, r_{t+h}), \quad t\text{-stat} = IC\sqrt{\frac{n-2}{1-IC^2}}$$

**A/B Test (Welch t-test on quarterly Sharpes):**
$$H_0: \mu_{treatment} = \mu_{control}, \quad H_1: \mu_{treatment} > \mu_{control}$$

Power analysis: $n_{min} = 2\left(\frac{z_{\alpha/2} + z_{\beta}}{\delta}\right)^2$ with $\delta=0.3$, $\alpha=0.05$, $\beta=0.2$

---

## Architecture

```
yfinance (100 tickers, 10yr OHLCV + fundamentals)
              │
              ▼
┌─────────────────────────────────────────────┐
│         Apache Airflow DAG                   │
│  Schedule: 6:30 AM ET, Mon-Fri              │
│  Tasks: Ingest → Regime → Factors → QC →   │
│         Backtest → Cache → Alert            │
└──────────────────┬──────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   AWS S3       Snowflake    MLflow
 (raw CSVs)   (structured)  (experiments)
       │           │
       └─────┬─────┘
             ▼
┌─────────────────────────────────────────────┐
│         Factor Computation Engine            │
│  Value: P/B, P/E, P/S (inverted, z-scored) │
│  Momentum: 12-1 month return (skip 1mo)     │
│  Quality: ROE, margins, leverage             │
│  HMM: bull/bear/neutral regime states       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│      Walk-Forward Backtesting Engine         │
│  Train: 252 days | Test: 63 days | Rolling  │
│  Treatment: factor-weighted long-short       │
│  Control: equal-weight benchmark             │
│  A/B Test: Welch t-test on quarterly Sharpe │
│  Power analysis: min_n computed first        │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
    FastAPI REST       Tableau Public
    /factors/{ticker}  Live Dashboard
    /backtest/{strat}  (public URL)
    /ab-test
    /regime
```

---

## Production Infrastructure

| Component | Technology | Cost |
|---|---|---|
| Orchestration | Apache Airflow (Docker / Astronomer free tier) | Free |
| Data Warehouse | Snowflake (30-day trial → pay-per-query) | ~$0 idle |
| Raw Storage | AWS S3 (free tier 5GB) | ~$0 |
| Experiment Tracking | MLflow (local or remote) | Free |
| API | FastAPI on AWS EC2 t2.micro | ~$0 (free tier) |
| Dashboard | Tableau Public | Free (public) |
| Data | yfinance (free) | Free |

**Cost when idle:** $0. **Cost to run for demo:** < $0.50 (Snowflake query + EC2 minutes).

---

## Key Results (populated after first full run)

| Metric | Value |
|---|---|
| Universe | 100 S&P 500 constituents |
| Lookback | 10 years daily data |
| Factor IC (composite, bull regime) | TBD after run |
| Factor IC (composite, bear regime) | TBD after run |
| A/B test p-value (treatment vs control) | TBD after run |
| Walk-forward Sharpe (composite) | TBD after run |
| Walk-forward Sharpe (equal-weight) | TBD after run |
| Max drawdown (composite) | TBD after run |

---

## Setup

```bash
# 1. Clone and configure
cp .env.template .env
# Fill in SNOWFLAKE_*, AWS_*, MLFLOW_TRACKING_URI

# 2. Initialize Snowflake schema
# Connect to Snowflake and run: data/snowflake_schema.sql

# 3. Start local infrastructure
docker compose up -d  # Starts Airflow, MLflow, API

# 4. Create venv and install
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Run demo (no cloud required)
python main.py --mode demo

# 6. Run full pipeline
python main.py --mode full

# 7. Start API only
python main.py --mode api

# 8. Run tests
pytest tests/ -v
```

---

## References

Fama, E.F. & French, K.R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.

Asness, C., Frazzini, A. & Pedersen, L.H. (2019). Quality minus junk. *Review of Accounting Studies*, 24(1), 34–112.

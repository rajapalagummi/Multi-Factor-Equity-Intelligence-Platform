"""
QuantEdge — Data Ingestion Pipeline
Fetches OHLCV price data and fundamental ratios for S&P 500 universe
via yfinance free tier. Stores raw data to AWS S3 and Snowflake.
"""
import os
import logging
import json
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import boto3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# S&P 500 representative universe — 100 liquid tickers across sectors
UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "USB", "PNC",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "LLY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT", "HD",
    # Industrials
    "GE", "HON", "MMM", "CAT", "BA", "LMT", "RTX", "DE", "EMR", "ETN",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO", "PSX", "HAL",
    # Materials
    "LIN", "APD", "ECL", "DD", "NEM", "FCX", "ALB", "MOS", "CF", "FMC",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "WELL", "DLR", "AVB", "EQR",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES",
    # Communication
    "VZ", "T", "TMUS", "NFLX", "DIS", "CMCSA", "CHTR", "ATVI", "EA", "TTWO",
]

LOOKBACK_YEARS = 10


def fetch_price_data(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 252:
            logger.warning(f"Insufficient price data for {ticker}")
            return None
        df = df.reset_index()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
              for c in df.columns]
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        return None


def fetch_fundamentals(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "fetched_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Fundamentals fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "fetched_at": datetime.utcnow().isoformat()}


def upload_to_s3(df: pd.DataFrame, key: str, bucket: str) -> bool:
    try:
        s3 = boto3.client("s3")
        csv_buffer = df.to_csv(index=False)
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)
        logger.info(f"Uploaded {len(df)} rows to s3://{bucket}/{key}")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed for {key}: {e}")
        return False


def upload_to_snowflake(df: pd.DataFrame, table: str) -> bool:
    try:
        import snowflake.connector
        from snowflake.connector.pandas_tools import write_pandas

        conn = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        )
        df.columns = [c.upper() for c in df.columns]
        success, nchunks, nrows, _ = write_pandas(conn, df, table.upper(),
                                                   auto_create_table=True,
                                                   overwrite=False)
        conn.close()
        logger.info(f"Wrote {nrows} rows to Snowflake {table}")
        return success
    except Exception as e:
        logger.error(f"Snowflake upload failed for {table}: {e}")
        return False


def run_ingestion(tickers: list = None, years: int = LOOKBACK_YEARS):
    if tickers is None:
        tickers = UNIVERSE

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    bucket = os.environ.get("AWS_S3_BUCKET", "quantedge-data")

    logger.info(f"Ingesting {len(tickers)} tickers from {start_date} to {end_date}")

    all_prices = []
    all_fundamentals = []

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Processing {ticker}")

        prices = fetch_price_data(ticker, start_date, end_date)
        if prices is not None:
            all_prices.append(prices)

        fundamentals = fetch_fundamentals(ticker)
        all_fundamentals.append(fundamentals)

    if all_prices:
        price_df = pd.concat(all_prices, ignore_index=True)
        date_str = datetime.today().strftime("%Y%m%d")
        upload_to_s3(price_df, f"raw/prices/{date_str}/prices.csv", bucket)
        upload_to_snowflake(price_df, "PRICES_RAW")
        logger.info(f"Price ingestion complete: {len(price_df)} rows")

    if all_fundamentals:
        fund_df = pd.DataFrame(all_fundamentals)
        upload_to_s3(fund_df, f"raw/fundamentals/{date_str}/fundamentals.csv", bucket)
        upload_to_snowflake(fund_df, "FUNDAMENTALS_RAW")
        logger.info(f"Fundamentals ingestion complete: {len(fund_df)} rows")

    return len(all_prices), len(all_fundamentals)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n_prices, n_funds = run_ingestion()
    print(f"Ingested price data for {n_prices} tickers, fundamentals for {n_funds} tickers")

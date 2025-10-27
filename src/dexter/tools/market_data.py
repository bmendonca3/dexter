import datetime as dt
from typing import Literal, Optional

import pandas as pd
import yfinance as yf
from langchain.tools import tool
from pydantic import BaseModel, Field, validator

from dexter.utils.cache import is_offline, load_cache, save_cache


class PriceHistoryInput(BaseModel):
    """Input schema for the get_price_history tool."""

    ticker: str = Field(..., description="Public equity ticker symbol, e.g., 'AAPL'.")
    period: Literal["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"] = Field(
        "1y",
        description="Window of history to download from Yahoo Finance.",
    )
    interval: Literal["1d", "1wk", "1mo", "1h", "1m"] = Field(
        "1d", description="Sampling interval for price candles."
    )
    include_adj_close: bool = Field(
        True,
        description="Whether to include the adjusted close column in the response.",
    )
    end_date: Optional[str] = Field(
        None,
        description="Optional inclusive cutoff date (YYYY-MM-DD). Only data on/before this date is returned.",
    )

    @validator("ticker")
    def _normalize_ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Ticker cannot be empty.")
        return value

    @validator("end_date")
    def _validate_end_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        try:
            pd.Timestamp(value)
        except Exception as exc:
            raise ValueError("end_date must be a valid date string in YYYY-MM-DD format.") from exc
        return value


@tool(args_schema=PriceHistoryInput)
def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    include_adj_close: bool = True,
    end_date: Optional[str] = None,
) -> dict:
    """
    Download historical OHLCV data for the requested ticker from Yahoo Finance.

    Returns a dictionary with metadata and a list of bars so downstream tools can
    easily consume the series.
    """
    cache_key = f"{ticker}_{period}_{interval}_{end_date or 'latest'}"
    cached_payload = load_cache("price_history", cache_key) if is_offline() else None
    if cached_payload is not None:
        cached_payload.setdefault("source", "cache")
        return cached_payload

    if is_offline():
        raise RuntimeError(
            "Price history unavailable in offline mode — no cached data found for the requested parameters."
        )

    end_param = None
    if end_date:
        end_ts = pd.Timestamp(end_date).normalize()
        # yfinance treats `end` as exclusive, so add a day to include the target date.
        end_param = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        end=end_param,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return {
            "resource": "price_history",
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "bars": [],
            "message": "No price data returned for the requested parameters.",
        }

    if isinstance(data.columns, pd.MultiIndex):
        flattened = []
        for col in data.columns:
            if isinstance(col, tuple):
                non_empty = [c for c in col if c]
                flattened.append(non_empty[0] if non_empty else "value")
            else:
                flattened.append(col)
        data.columns = flattened

    column_aliases = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
    }
    normalized_columns = {}
    for col in data.columns:
        key = str(col).strip()
        canonical = column_aliases.get(key.lower(), key)
        normalized_columns[col] = canonical
    data = data.rename(columns=normalized_columns)

    data = data.reset_index()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"]).dt.tz_localize(None)
        data["date_str"] = data["Date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        # Fallback for extended data formats
        data["date_str"] = pd.Series(data.index).apply(
            lambda x: pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S")
        )

    bars = []
    for _, row in data.iterrows():
        date_value = row.get("date_str")
        if isinstance(date_value, pd.Series):
            # In rare cases (multi-index columns) ensure we extract scalar
            date_value = date_value.iloc[0]
        if not isinstance(date_value, str):
            date_value = pd.Timestamp(date_value).strftime("%Y-%m-%d %H:%M:%S")
        bar = {
            "date": date_value,
            "open": float(row.get("Open", 0.0) or 0.0),
            "high": float(row.get("High", 0.0) or 0.0),
            "low": float(row.get("Low", 0.0) or 0.0),
            "close": float(row.get("Close", 0.0) or 0.0),
            "volume": float(row.get("Volume", 0.0) or 0.0),
        }
        if include_adj_close and "Adj Close" in data.columns:
            bar["adj_close"] = float(row["Adj Close"])
        bars.append(bar)

    payload = {
        "resource": "price_history",
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "bars": bars,
        "latest_close": bars[-1]["close"],
        "latest_date": bars[-1]["date"],
        "downloaded_at": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if end_date:
        payload["end_date"] = end_date

    save_cache("price_history", cache_key, payload)
    payload.setdefault("source", "live")
    return payload

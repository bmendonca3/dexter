from typing import Any, Dict, List

import pandas as pd
import yfinance as yf
from langchain.tools import tool
from pydantic import BaseModel, Field, validator

from dexter.utils.cache import is_offline, load_cache, save_cache


def _safe_to_dict(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    if frame is None or isinstance(frame, float) or frame is ...:
        return []
    try:
        frame = frame.fillna(value=None)
        frame = frame.reset_index()
    except Exception:
        return []
    records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        record: Dict[str, Any] = {}
        for key, value in row.items():
            key_str = str(key)
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = value
            if isinstance(value, pd.Timestamp):
                value = value.strftime("%Y-%m-%d")
            record[key_str] = value
        records.append(record)
    return records


class FinancialSnapshotInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., 'MSFT'.")

    @validator("ticker")
    def _clean_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty.")
        return cleaned


@tool(args_schema=FinancialSnapshotInput)
def get_financial_snapshot(ticker: str) -> dict:
    """
    Retrieve key fundamental data for a ticker using Yahoo Finance, including
    valuation ratios, balance sheet line items, income statement summaries,
    and cash flow data where available.
    """
    cache_key = ticker
    cached_payload = load_cache("financial_snapshot", cache_key) if is_offline() else None
    if cached_payload is not None:
        cached_payload.setdefault("source", "cache")
        return cached_payload

    if is_offline():
        raise RuntimeError(
            "Financial snapshot unavailable in offline mode — no cached data found for the requested ticker."
        )

    instrument = yf.Ticker(ticker)

    info = instrument.info or {}
    fast_info = {}
    try:
        fast_info = instrument.fast_info or {}
    except Exception:
        fast_info = {}

    income_statement = _safe_to_dict(instrument.income_stmt)
    balance_sheet = _safe_to_dict(instrument.balance_sheet)
    cash_flow = _safe_to_dict(instrument.cashflow)

    financials = {
        "resource": "financial_snapshot",
        "ticker": ticker,
        "company_name": info.get("longName") or info.get("shortName"),
        "currency": info.get("financialCurrency"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "price_to_book": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "52_week_high": fast_info.get("yearHigh"),
        "52_week_low": fast_info.get("yearLow"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "income_statement": income_statement,
        "balance_sheet": balance_sheet,
        "cash_flow": cash_flow,
    }

    save_cache("financial_snapshot", cache_key, financials)
    financials.setdefault("source", "live")
    return financials

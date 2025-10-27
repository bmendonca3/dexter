#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from dexter.agent import Agent


load_dotenv()


def _download_price_history(symbol: str, start: str, end: str) -> Tuple[pd.DataFrame, str]:
    """Download daily prices with a buffer so we can look up month-end closes."""
    data = yf.download(
        symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"No price history returned for {symbol}.")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    return data[[price_column]].rename(columns={price_column: "close"}), price_column


def _price_on_or_before(df: pd.DataFrame, target: pd.Timestamp) -> Tuple[float, pd.Timestamp]:
    """Return the closing price on or immediately before the requested date."""
    eligible = df.loc[:target]
    if eligible.empty:
        raise RuntimeError(f"No trading data available on or before {target.date()}.")
    price = float(eligible["close"].iloc[-1])
    return price, eligible.index[-1]


def simulate_nvda(from_month: str = "2024-08-31", through: str = None) -> Dict[str, object]:
    """
    Simulate monthly NVDA plans starting at `from_month` and measure returns to the latest close.
    Writes detailed outputs to JSON and CSV for later inspection.
    """
    price_df, _ = _download_price_history("NVDA", start="2024-07-01", end="2025-10-27")
    latest_price = float(price_df["close"].iloc[-1])
    latest_date = price_df.index[-1]

    through_date = pd.Timestamp(through) if through else latest_date
    through_date = min(through_date, latest_date)

    evaluation_dates = pd.date_range(from_month, through_date, freq="ME")
    results: List[Dict[str, object]] = []

    for evaluation_date in evaluation_dates:
        if evaluation_date > through_date:
            break
        ref_price, ref_trading_day = _price_on_or_before(price_df, evaluation_date)
        os.environ["DEXTER_END_DATE"] = ref_trading_day.strftime("%Y-%m-%d")

        agent = Agent()
        query = (
            f"As of {ref_trading_day.strftime('%Y-%m-%d')}, build a long plan for NVDA with strong risk controls. "
            "Use only data available up to that date."
        )
        analysis = agent.run(query)

        realized_return = (latest_price / ref_price) - 1
        results.append(
            {
                "as_of_requested": evaluation_date.strftime("%Y-%m-%d"),
                "as_of_trading_day": ref_trading_day.strftime("%Y-%m-%d"),
                "close_price": round(ref_price, 4),
                "latest_price": round(latest_price, 4),
                "return_since_as_of": realized_return,
                "return_pct": realized_return * 100,
                "analysis": analysis,
            }
        )

    os.environ.pop("DEXTER_END_DATE", None)

    if not results:
        raise RuntimeError("Simulation produced no results.")

    detail_path = Path("simulated_nvda_trades.json")
    detail_path.write_text(json.dumps(results, indent=2))

    summary = {
        "latest_price_date": latest_date.strftime("%Y-%m-%d"),
        "latest_price": round(latest_price, 4),
        "total_periods": len(results),
        "average_return_pct": sum(r["return_pct"] for r in results) / len(results),
        "positive_return_periods": sum(1 for r in results if r["return_since_as_of"] > 0),
    }
    summary_path = Path("simulated_nvda_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    report = simulate_nvda()
    print(json.dumps(report, indent=2))

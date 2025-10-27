import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.tools import tool
from pydantic import BaseModel, Field, validator

from dexter.utils.cache import is_offline, load_cache, save_cache


TRADING_DAYS = 252


@dataclass
class StrategyMetrics:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    hit_rate: float
    avg_gain: float
    avg_loss: float
    exposure: float

    def to_dict(self) -> Dict[str, float]:
        cleaned: Dict[str, float] = {}
        for key, value in asdict(self).items():
            if value != value or value is None:  # NaN or None
                cleaned[key] = 0.0
            else:
                cleaned[key] = round(float(value), 4)
        return cleaned


def _annualized_return(series: pd.Series) -> float:
    cumulative_return = (1 + series).prod()
    periods = series.size
    if periods == 0 or cumulative_return <= 0:
        return 0.0
    return cumulative_return ** (TRADING_DAYS / periods) - 1


def _annualized_vol(series: pd.Series) -> float:
    if series.std() == 0:
        return 0.0
    return series.std() * math.sqrt(TRADING_DAYS)


def _downside_vol(series: pd.Series) -> float:
    downside = series[series < 0]
    if downside.empty:
        return 0.0
    return downside.std() * math.sqrt(TRADING_DAYS)


def _max_drawdown(cumulative: pd.Series) -> float:
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return drawdowns.min()


def _hit_rate(series: pd.Series) -> float:
    gains = (series > 0).sum()
    losses = (series < 0).sum()
    total = gains + losses
    if total == 0:
        return 0.0
    return gains / total


def _average_gain_loss(series: pd.Series) -> tuple[float, float]:
    gains = series[series > 0]
    losses = series[series < 0]
    avg_gain = gains.mean() if not gains.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    return avg_gain, avg_loss


class LongStrategyInput(BaseModel):
    ticker: str = Field(..., description="Ticker to analyze, e.g., AAPL")
    benchmark: str = Field(
        "SPY",
        description="Ticker used as a benchmark for relative performance.",
    )
    lookback_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of years of history to use for analysis.",
    )
    short_window: int = Field(
        21,
        ge=5,
        le=120,
        description="Short moving average window (trading days).",
    )
    long_window: int = Field(
        63,
        ge=20,
        le=252,
        description="Long moving average window (trading days).",
    )
    risk_free_rate: float = Field(
        0.02,
        ge=0.0,
        le=0.1,
        description="Annualized risk free rate used for Sharpe calculations.",
    )
    end_date: Optional[str] = Field(
        None,
        description="Optional inclusive cutoff date (YYYY-MM-DD) for the historical analysis.",
    )

    @validator("ticker", "benchmark")
    def _normalize_symbols(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty.")
        return cleaned

    @validator("long_window")
    def _validate_windows(cls, long_window: int, values):
        short_window = values.get("short_window", 21)
        if short_window >= long_window:
            raise ValueError("long_window must be greater than short_window.")
        return long_window

    @validator("end_date")
    def _validate_end_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        try:
            pd.Timestamp(value)
        except Exception as exc:
            raise ValueError("end_date must be a valid date string in YYYY-MM-DD format.") from exc
        return value


def _download_history(symbol: str, years: int, end_date: Optional[str]) -> pd.DataFrame:
    period = f"{years}y" if years < 10 else "max"
    cache_key = f"{symbol}_{period}_{end_date or 'latest'}"
    cached_payload = load_cache("strategy_history", cache_key) if is_offline() else None
    if cached_payload is not None:
        series_data = cached_payload.get("series", [])
        if not series_data:
            raise ValueError(f"No cached time series found for {symbol}.")
        series = pd.Series(
            {pd.Timestamp(entry["date"]): float(entry["price"]) for entry in series_data}
        )
        series.index = pd.to_datetime(series.index)
        series.sort_index(inplace=True)
        return series.rename(symbol).to_frame()

    end_param = None
    if end_date:
        end_ts = pd.Timestamp(end_date).normalize()
        end_param = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    if is_offline():
        raise ValueError(
            f"Strategy history unavailable in offline mode — no cached data for {symbol} ({period})."
        )

    data = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        end=end_param,
    )
    if data.empty:
        raise ValueError(f"No historical prices returned for {symbol}.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) and col else col for col in data.columns]
    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    price_series = data[price_column].astype(float)
    price_series.index = pd.to_datetime(price_series.index)
    price_series.sort_index(inplace=True)
    price_series = price_series.rename(symbol)

    payload = {
        "symbol": symbol,
        "period": period,
        "end_date": end_date,
        "series": [
            {"date": idx.strftime("%Y-%m-%d"), "price": float(value)}
            for idx, value in price_series.items()
        ],
    }
    save_cache("strategy_history", cache_key, payload)

    return price_series.to_frame()


def _compute_strategy(prices: pd.Series, short_window: int, long_window: int) -> pd.DataFrame:
    frame = pd.DataFrame({"price": prices})
    frame["return"] = frame["price"].pct_change()
    frame["ma_short"] = frame["price"].rolling(short_window).mean()
    frame["ma_long"] = frame["price"].rolling(long_window).mean()
    frame["signal"] = (frame["ma_short"] > frame["ma_long"]).astype(int)
    frame["position"] = frame["signal"].shift(1).fillna(0)
    frame["strategy_return"] = frame["position"] * frame["return"]
    frame = frame.dropna()
    return frame


def _summarize_strategy(strategy_frame: pd.DataFrame, risk_free_rate: float) -> StrategyMetrics:
    returns = strategy_frame["strategy_return"]
    cagr = _annualized_return(returns)
    vol = _annualized_vol(returns)
    downside_vol = _downside_vol(returns)
    sharpe = (cagr - risk_free_rate) / vol if vol > 0 else 0.0
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
    cumulative = (1 + returns).cumprod()
    max_dd = _max_drawdown(cumulative)
    hit_rate = _hit_rate(returns)
    avg_gain, avg_loss = _average_gain_loss(returns)
    exposure = strategy_frame["position"].mean()
    return StrategyMetrics(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        avg_gain=avg_gain,
        avg_loss=avg_loss,
        exposure=exposure,
    )


@tool(args_schema=LongStrategyInput)
def evaluate_long_strategy(
    ticker: str,
    benchmark: str = "SPY",
    lookback_years: int = 3,
    short_window: int = 21,
    long_window: int = 63,
    risk_free_rate: float = 0.02,
    end_date: Optional[str] = None,
) -> dict:
    """
    Evaluate a long-only moving-average crossover strategy for the given ticker.
    Returns performance metrics, current signal state, and risk stats to help
    the agent decide on long allocations.
    """
    ticker_history = _download_history(ticker, lookback_years, end_date)
    benchmark_history = _download_history(benchmark, lookback_years, end_date)

    joined = ticker_history.join(benchmark_history, how="inner")
    joined = joined.dropna()
    if joined.empty:
        raise ValueError("Not enough overlapping history between ticker and benchmark.")

    prices = joined[ticker]
    benchmark_returns = joined[benchmark].pct_change().dropna()

    strategy_frame = _compute_strategy(prices, short_window, long_window)
    strategy_metrics = _summarize_strategy(strategy_frame, risk_free_rate)
    buy_and_hold_metrics = _summarize_strategy(
        strategy_frame.assign(strategy_return=strategy_frame["return"]),
        risk_free_rate,
    )

    benchmark_metrics = _summarize_strategy(
        strategy_frame.assign(strategy_return=benchmark_returns.reindex(strategy_frame.index).fillna(0)),
        risk_free_rate,
    )

    latest_row = strategy_frame.iloc[-1]
    signal_state = "long" if latest_row["position"] > 0 else "flat"

    risk_snapshot = {
        "current_signal": signal_state,
        "last_price": float(latest_row["price"]),
        "ma_short": float(latest_row["ma_short"]),
        "ma_long": float(latest_row["ma_long"]),
        "distance_from_long_ma_pct": float((latest_row["price"] / latest_row["ma_long"]) - 1),
        "12m_volatility": float(strategy_frame["return"].std() * math.sqrt(TRADING_DAYS)),
        "max_drawdown": strategy_metrics.max_drawdown,
    }

    recommendation = "monitor"
    if signal_state == "long":
        if risk_snapshot["distance_from_long_ma_pct"] < 0.05 and strategy_metrics.sharpe > 1:
            recommendation = "scale_up"
        else:
            recommendation = "hold"
    else:
        if strategy_frame["ma_short"].iloc[-2] <= strategy_frame["ma_long"].iloc[-2] and latest_row["ma_short"] > latest_row["ma_long"]:
            recommendation = "prepare_entry"

    return {
        "resource": "long_strategy_analysis",
        "ticker": ticker,
        "benchmark": benchmark,
        "lookback_years": lookback_years,
        "parameters": {
            "short_window": short_window,
            "long_window": long_window,
            "risk_free_rate": risk_free_rate,
        },
        "end_date": end_date,
        "strategy_metrics": strategy_metrics.to_dict(),
        "buy_and_hold_metrics": buy_and_hold_metrics.to_dict(),
        "benchmark_metrics": benchmark_metrics.to_dict(),
        "risk_snapshot": risk_snapshot,
        "latest_signal_date": strategy_frame.index[-1].strftime("%Y-%m-%d"),
        "recommendation": recommendation,
    }

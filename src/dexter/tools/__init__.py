from typing import Any, Callable, List

from dexter.tools.fundamentals import get_financial_snapshot
from dexter.tools.long_strategy import evaluate_long_strategy
from dexter.tools.market_data import get_price_history

TOOLS: List[Callable[..., Any]] = [
    get_price_history,
    get_financial_snapshot,
    evaluate_long_strategy,
]

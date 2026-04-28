from dataclasses import dataclass
from enum import Enum


class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass
class GapSignal:
    stock_code: str
    exchange_code: str
    prev_close: float
    today_open: float
    gap_pct: float
    direction: TradeDirection


@dataclass
class TradeResult:
    signal: GapSignal
    entry_price: float
    target: float
    stop_loss: float
    order_response: dict

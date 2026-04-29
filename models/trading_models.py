from dataclasses import dataclass
from enum import Enum
from datetime import date


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


@dataclass
class BacktestTradeResult:
    trade_date: date
    direction: TradeDirection
    prev_close: float
    entry_price: float
    target: float
    stop_loss: float
    exit_price: float
    exit_reason: str   # "target", "stop_loss", or "close"
    pnl: float
    gap_pct: float

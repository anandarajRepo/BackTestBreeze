from dataclasses import dataclass
from enum import Enum
from datetime import date


class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"
    NONE = "none"


class GapType(str, Enum):
    FULL_GAP_UP = "full_gap_up"
    PARTIAL_GAP_UP = "partial_gap_up"
    FULL_GAP_DOWN = "full_gap_down"
    PARTIAL_GAP_DOWN = "partial_gap_down"
    NONE = "none"


@dataclass
class GapBehaviourStats:
    sample_count: int
    continuation_count: int
    reversal_count: int
    continuation_rate: float  # percentage 0–100
    reversal_rate: float      # percentage 0–100


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
    gap_type: GapType
    prev_close: float
    entry_price: float
    target: float
    stop_loss: float
    exit_price: float
    exit_reason: str   # "target", "stop_loss", or "close"
    pnl: float
    gap_pct: float
    continuation_rate: float
    reversal_rate: float
    gap_history_count: int

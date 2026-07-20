from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class ROCSignalDirection(str, Enum):
    BUY  = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass
class ROCSignal:
    direction: ROCSignalDirection
    signal_idx: int          # index of the signal candle within the day's candle list
    signal_time: str         # datetime of the crossover candle
    roc_value: float         # ROC % at the crossover candle
    entry_price: float       # close of the crossover candle


@dataclass
class ROCTradeResult:
    trade_date: date
    direction: ROCSignalDirection
    roc_value: float
    entry_price: float
    target: float
    stop_loss: float
    exit_price: float
    exit_reason: str            # "target" | "stop_loss" | "trailing_stop" | "breakeven" | "close"
    pnl: float
    signal_time: str

    momentum_score: Optional[float] = None
    momentum_rank: Optional[int]    = None

    partial_exit_price: Optional[float] = None
    partial_exit_qty: int               = 0

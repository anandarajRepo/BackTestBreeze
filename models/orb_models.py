from dataclasses import dataclass
from datetime import date
from enum import Enum


class BreakoutDirection(str, Enum):
    BUY  = "buy"
    SELL = "sell"
    NONE = "none"


@dataclass
class OpenRange:
    high: float
    low: float

    @property
    def range_size(self) -> float:
        return round(self.high - self.low, 2)


@dataclass
class ORBTradeResult:
    trade_date: date
    direction: BreakoutDirection
    orb_high: float
    orb_low: float
    entry_price: float
    target: float
    stop_loss: float
    exit_price: float
    exit_reason: str   # "target" | "stop_loss" | "close"
    pnl: float
    breakout_time: str

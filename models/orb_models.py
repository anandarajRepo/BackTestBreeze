from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


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
    exit_reason: str            # "target" | "stop_loss" | "close"
    pnl: float
    breakout_time: str

    # Momentum & trend fields (populated when filters are enabled)
    momentum_score: Optional[float] = None      # composite 0–100
    trend_direction: Optional[str]  = None      # "UPTREND" | "DOWNTREND" | "SIDEWAYS"
    trend_strength: Optional[float] = None      # 0–100
    intraday_trend: Optional[str]   = None      # intraday direction for that day

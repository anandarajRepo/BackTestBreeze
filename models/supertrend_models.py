from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class PartialExit:
    """A single partial (or final) exit leg of a Supertrend position."""
    time: datetime
    price: float
    shares: int
    pnl: float
    reason: str               # "TP_25", "TP_50", "TARGET", "TRAILING_STOP",
                              # "BREAKEVEN", "SUPERTREND_FLIP", "SQUARE_OFF"


@dataclass
class SuperTrendTradeResult:
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime       # time of the final leg that closed the position
    entry_price: float
    exit_price: float         # share-weighted average exit price across all legs
    shares: int               # total shares entered
    pnl: float                # total pnl across all partial + final legs
    exit_reason: str          # reason of the final leg
    supertrend_at_entry: float
    supertrend_at_exit: float
    atr_at_entry: float
    duration_minutes: int
    partials: list[PartialExit] = field(default_factory=list)


@dataclass
class WeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[SuperTrendTradeResult] = field(default_factory=list)
    pe_trades: list[SuperTrendTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[SuperTrendTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class SymbolMetrics:
    symbol: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    profit_factor: float
    best_trade: float
    worst_trade: float
    avg_duration_minutes: float
    max_consecutive_losses: int

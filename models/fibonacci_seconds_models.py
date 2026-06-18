from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class FibSecondsTradeResult:
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    exit_reason: str          # "TARGET" | "STOP_LOSS" | "TRAILING_STOP" | "BREAKEVEN" | "PARTIAL_BOOK" | "SQUARE_OFF"
    swing_high: float
    swing_low: float
    fib_entry_level: float     # the Fibonacci retracement price the entry was taken at
    fib_ratio: float           # the Fibonacci ratio of the entry level (e.g. 0.618)
    rsi_at_entry: float
    breakout_volume: float
    avg_volume: float
    volume_ratio: float
    duration_minutes: int


@dataclass
class WeeklyFibExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[FibSecondsTradeResult] = field(default_factory=list)
    pe_trades: list[FibSecondsTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[FibSecondsTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class FibSymbolMetrics:
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

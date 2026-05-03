from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class EMATradeResult:
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
    exit_reason: str          # "EMA_CROSS" or "SQUARE_OFF"
    fast_ema_at_entry: float
    slow_ema_at_entry: float
    fast_ema_at_exit: float
    slow_ema_at_exit: float
    duration_minutes: int


@dataclass
class EMAWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[EMATradeResult] = field(default_factory=list)
    pe_trades: list[EMATradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[EMATradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class EMASymbolMetrics:
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

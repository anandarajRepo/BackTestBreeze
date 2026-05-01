from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class BBTradeResult:
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
    exit_reason: str          # "BB_REVERSAL" or "SQUARE_OFF"
    upper_band_at_entry: float
    middle_band_at_entry: float
    lower_band_at_entry: float
    upper_band_at_exit: float
    middle_band_at_exit: float
    lower_band_at_exit: float
    duration_minutes: int


@dataclass
class BBWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[BBTradeResult] = field(default_factory=list)
    pe_trades: list[BBTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[BBTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class BBSymbolMetrics:
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

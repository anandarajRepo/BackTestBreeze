from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class HTFTradeResult:
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
    exit_reason: str          # "SQUARE_OFF" or "EOD"
    htf_bias: str             # "BULLISH" or "BEARISH"
    htf_open: float
    htf_close: float
    ema_at_entry: float
    volume_at_entry: float
    duration_minutes: int


@dataclass
class WeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[HTFTradeResult] = field(default_factory=list)
    pe_trades: list[HTFTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[HTFTradeResult]:
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

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class HeikinAshiSupertrendTradeResult:
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
    exit_reason: str          # "TARGET" | "STOP_LOSS" | "SQUARE_OFF"
    stop_loss: float
    target: float
    adx_at_entry: float
    duration_minutes: int


@dataclass
class WeeklyHeikinAshiSupertrendExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[HeikinAshiSupertrendTradeResult] = field(default_factory=list)
    pe_trades: list[HeikinAshiSupertrendTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[HeikinAshiSupertrendTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class HeikinAshiSupertrendSymbolMetrics:
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

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class ORBSecondsTradeResult:
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
    exit_reason: str          # "TARGET" | "STOP_LOSS" | "TRAILING_STOP" | "SQUARE_OFF"
    orb_high: float
    orb_low: float
    breakout_volume: float
    orb_avg_volume: float
    volume_ratio: float
    duration_minutes: int


@dataclass
class WeeklyORBExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[ORBSecondsTradeResult] = field(default_factory=list)
    pe_trades: list[ORBSecondsTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[ORBSecondsTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class ORBSymbolMetrics:
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

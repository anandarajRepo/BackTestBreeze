from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class RSITradeResult:
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
    exit_reason: str          # "RSI_NEUTRAL", "SQUARE_OFF"
    signal_type: str          # "DIVERGENCE" or "CONVERGENCE"
    rsi_at_entry: float
    rsi_at_exit: float
    price_at_entry: float
    price_at_exit: float
    duration_minutes: int


@dataclass
class RSIWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[RSITradeResult] = field(default_factory=list)
    pe_trades: list[RSITradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[RSITradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class RSISymbolMetrics:
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

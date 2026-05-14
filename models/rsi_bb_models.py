from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class RSIBBTradeResult:
    symbol: str
    option_type: str           # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    exit_reason: str           # "BB_UPPER_EXIT", "BB_LOWER_EXIT", "SQUARE_OFF"
    rsi_at_entry: float
    bb_upper_at_entry: float
    bb_middle_at_entry: float
    bb_lower_at_entry: float
    rsi_at_exit: float
    bb_upper_at_exit: float
    bb_middle_at_exit: float
    bb_lower_at_exit: float
    duration_minutes: int


@dataclass
class RSIBBWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[RSIBBTradeResult] = field(default_factory=list)
    pe_trades: list[RSIBBTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[RSIBBTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class RSIBBSymbolMetrics:
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

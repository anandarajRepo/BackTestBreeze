from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class ERTradeResult:
    symbol: str
    option_type: str          # "CE" (long/green regime) or "PE" (short/red regime)
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    exit_reason: str          # "TARGET" | "STOP_LOSS" | "BREAKEVEN" | "REGIME_SHIFT" | "EOD_FLATTEN"
    direction: str            # "long" | "short"
    entry_mode: str           # the entry mode that produced the trade
    regime_at_entry: str      # "green" | "red"
    er_at_entry: float        # Efficiency Ratio on the signal candle
    duration_minutes: int


@dataclass
class ERWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[ERTradeResult] = field(default_factory=list)
    pe_trades: list[ERTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[ERTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class ERSymbolMetrics:
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

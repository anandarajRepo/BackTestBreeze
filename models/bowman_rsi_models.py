from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class BowmanRSITradeResult:
    """
    Result of a single Bowman RSI long position on an option premium.

    The position is opened when the fast RSI crosses up through the oversold
    level (confirmed by the higher-timeframe RSI and EMA filters) and closed
    in a single leg by the configured exit mode (ATR trailing stop, bearish
    divergence + pivot high, pivot high only, or fixed percentages), or by
    the mandatory square-off.
    """
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
    exit_reason: str
    rsi_at_entry: float
    htf_rsi_at_entry: float
    ema_at_entry: float       # HTF EMA value at entry (NaN if filter disabled)
    rsi_at_exit: float
    atr_at_entry: float
    duration_minutes: int


@dataclass
class BowmanRSIWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[BowmanRSITradeResult] = field(default_factory=list)
    pe_trades: list[BowmanRSITradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[BowmanRSITradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class BowmanRSISymbolMetrics:
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

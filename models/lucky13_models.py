from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Lucky13TradeResult:
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float        # option entry price
    exit_price: float         # option exit price
    shares: int
    pnl: float
    exit_reason: str          # "PROFIT_TARGET" | "STOP_LOSS" | "TRAIL_STOP" | "EMA_CROSS" | "SQUARE_OFF"
    # spot indicators at entry
    spot_close_at_entry: float
    ema13_at_entry: float
    vwap_at_entry: float
    volume_ratio_at_entry: float  # volume / vol_sma
    ema5m_at_entry: float
    # spot indicators at exit
    ema13_at_exit: float
    duration_minutes: int
    # filter states at entry
    volume_filter: bool
    vwap_filter: bool
    htf_ema_filter: bool


@dataclass
class Lucky13WeeklyResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[Lucky13TradeResult] = field(default_factory=list)
    pe_trades: list[Lucky13TradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[Lucky13TradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class Lucky13SymbolMetrics:
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

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class IBSecondsTradeResult:
    """
    Result of a single Initial Balance (IB) option trade.

    The Initial Balance is the price range (high/low) established during the
    first `ib_minutes` of the session. A trade is taken either as a BREAKOUT
    (the option premium breaks firmly above its IB high and trend continues) or
    as a REVERSAL / failed-breakdown trap (the premium dips below its IB low but
    fails to hold and rotates back up into the range). `entry_mode` records which
    of the two IB approaches produced the trade.
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
    exit_reason: str          # "TARGET" | "STOP_LOSS" | "TRAILING_STOP" | "BREAKEVEN" | "PARTIAL_BOOK" | "SQUARE_OFF"
    entry_mode: str           # "BREAKOUT" | "REVERSAL"
    ib_high: float
    ib_low: float
    breakout_volume: float
    ib_avg_volume: float
    volume_ratio: float
    duration_minutes: int


@dataclass
class WeeklyIBExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[IBSecondsTradeResult] = field(default_factory=list)
    pe_trades: list[IBSecondsTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[IBSecondsTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class IBSymbolMetrics:
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

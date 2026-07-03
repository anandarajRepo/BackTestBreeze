from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class VWAPBand3TradeResult:
    """
    Result of a single VWAP Band 3 mean-reversion position.

    A LONG position is opened when the option price touches the lower band 3
    and a SHORT position when it touches the upper band 3. The take-profit is
    the VWAP value at entry (fixed for the life of the trade) and the
    stop-loss sits the same distance on the other side of the entry (1:1 R:R).
    """
    symbol: str
    option_type: str          # "CE" or "PE"
    direction: str            # "LONG" or "SHORT"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    exit_reason: str          # "TARGET", "STOP_LOSS", "SQUARE_OFF"
    vwap_at_entry: float
    band_at_entry: float      # the band 3 level that triggered the entry
    target_price: float       # VWAP fixed at entry
    stop_price: float
    duration_minutes: int


@dataclass
class VWAPBand3WeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[VWAPBand3TradeResult] = field(default_factory=list)
    pe_trades: list[VWAPBand3TradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[VWAPBand3TradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class VWAPBand3SymbolMetrics:
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

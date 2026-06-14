from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class VWTFTradeResult:
    """
    Result of a single volume-weighted trend-following position.

    A position may be exited in up to three scale-out legs (25% / 25% / 50%)
    as price reaches 25%, 50% and 100% of the profit target, or closed in a
    single leg by a protective stop / reversal / square-off. `exit_price` is
    the share-weighted average exit price across all legs, `pnl` is the total
    realized profit/loss, and `exit_reason` reflects the reason for the final
    (position-closing) leg. `scale_out_legs` is a human-readable summary of the
    individual partial exits.
    """
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float         # share-weighted average exit price
    shares: int               # initial position size
    pnl: float                # total realized PnL across all scale-out legs
    exit_reason: str          # reason for the final closing leg
    vwap_at_entry: float
    ema_at_entry: float
    vwap_at_exit: float
    ema_at_exit: float
    target_price: float
    duration_minutes: int
    scale_out_legs: str = ""  # e.g. "TP1@12.5x25 | TP2@15.0x25 | TARGET@20.0x50"


@dataclass
class VWTFWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[VWTFTradeResult] = field(default_factory=list)
    pe_trades: list[VWTFTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[VWTFTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class VWTFSymbolMetrics:
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

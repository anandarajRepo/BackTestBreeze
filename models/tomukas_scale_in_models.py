from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class TomukasTradeResult:
    """
    Result of a single Tomukas Scale-In v2 position.

    A position is built in up to five scale-in legs (pyramiding), each
    triggered by a fresh liquidity-sweep signal while the trend filter holds.
    `entry_price` is the share-weighted average entry price across all
    scale-in legs, `shares` is the total accumulated position size, and
    `scale_in_legs` is a human-readable summary of the individual entries.
    The whole position is closed in one leg (ATR take-profit, square-off or
    end of data).
    """
    symbol: str
    option_type: str          # "CE" or "PE"
    strike: int
    expiry_date: date
    entry_time: datetime      # time of the FIRST scale-in leg
    exit_time: datetime
    entry_price: float        # share-weighted average entry price
    exit_price: float
    shares: int               # total accumulated position size
    pnl: float                # realized PnL for the whole position
    exit_reason: str
    ema_fast_at_entry: float  # EMA100 at first entry
    ema_slow_at_entry: float  # EMA200 at first entry
    atr_at_entry: float
    target_price: float       # TP level active at exit (avg + ATR × mult)
    num_entries: int          # number of scale-in legs filled
    duration_minutes: int
    scale_in_legs: str = ""   # e.g. "L1@12.5x100 | L2@10.0x100 | L3@8.0x200"


@dataclass
class TomukasWeeklyExpiryResult:
    expiry_date: date
    atm_strike: int
    nifty_open: float
    ce_trades: list[TomukasTradeResult] = field(default_factory=list)
    pe_trades: list[TomukasTradeResult] = field(default_factory=list)

    @property
    def all_trades(self) -> list[TomukasTradeResult]:
        return self.ce_trades + self.pe_trades


@dataclass
class TomukasSymbolMetrics:
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
